import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow.keras.backend as K  # 引入 Keras 后端用于写 Loss
import numpy as np
import os
import math 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random 

# -------------------------------------------------------------------------
# 1. 配置区域 (Configuration)
# -------------------------------------------------------------------------
# === 关键开关 ===
# True: 使用 Focal Loss (不传 class_weight)
# False: 使用原版 Binary Cross-Entropy (传入 class_weight)
USE_FOCAL_LOSS = True 

IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
INITIAL_LR = 1e-4  
FINE_TUNE_LR = 1e-5
EPOCHS_HEAD = 30       
EPOCHS_FINE_TUNE = 20

BASE_DATA_DIR = './dataset/' 
TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train')
VALID_DIR = os.path.join(BASE_DATA_DIR, 'validation')

# -------------------------------------------------------------------------
# 2. 固定随机种子
# -------------------------------------------------------------------------
def reset_random_seeds(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    print(f"Random seeds set to {seed_value}")

reset_random_seeds() 

# -------------------------------------------------------------------------
# 3. 自定义 Binary Focal Loss
# -------------------------------------------------------------------------
def binary_focal_loss(gamma=2., alpha=0.25):
    """
    Binary Focal Loss 实现
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        loss = -alpha_t * K.pow((1 - p_t), gamma) * K.log(p_t)
        return K.mean(loss, axis=1)
    
    return focal_loss_fixed

# -------------------------------------------------------------------------
# 4. 模型构建与工具函数
# -------------------------------------------------------------------------
def build_baseline_model(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)):
    base_model = ResNet50(weights='imagenet', 
                          include_top=False, 
                          input_shape=input_shape)
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x) 
    x = Dense(128, activation='relu')(x) 
    x = Dropout(0.5)(x) 
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def unfreeze_model(base_model, num_layers_to_unfreeze=30):
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    print(f"\n--- Unfreezing the top {num_layers_to_unfreeze} layers for fine-tuning ---")

def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,            
        rotation_range=15,       
        width_shift_range=0.1,     
        height_shift_range=0.1,    
        shear_range=0.1,           
        zoom_range=0.1,            
        horizontal_flip=True,      
        fill_mode='nearest'
    )
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary' 
    )
    validation_generator = validation_datagen.flow_from_directory(
        VALID_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False 
    )
    return train_generator, validation_generator

# Grad-CAM 相关函数
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out", pred_index=None):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_gradcam_visualization(img_array, heatmap, save_path="gradcam_result.jpg", alpha=0.4):
    img = np.uint8(255 * img_array[0])
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.image.resize(jet_heatmap[np.newaxis, ...], (img.shape[0], img.shape[1]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap[0])
    jet_heatmap = np.uint8(jet_heatmap * 255)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(save_path)
    print(f"Grad-CAM 结果已保存至: {save_path}")

# -------------------------------------------------------------------------
# 5. 主程序 (Main)
# -------------------------------------------------------------------------
def main():
    reset_random_seeds()
    
    # ------------------------------------------------------------------
    # A. 准备数据和计算权重
    # ------------------------------------------------------------------
    try:
        train_gen, valid_gen = get_data_generators()
        
        # 始终计算权重 (用于日志打印或原版 Loss)
        label_0 = train_gen.class_indices['Normal'] 
        label_1 = train_gen.class_indices['Scol'] 
        total_samples = train_gen.samples
        count_0 = np.sum(train_gen.labels == label_0)
        count_1 = np.sum(train_gen.labels == label_1)
        
        weight_0 = (1 / count_0) * (total_samples / 2.0)
        weight_1 = (1 / count_1) * (total_samples / 2.0)
        class_weights_dict = {label_0: weight_0, label_1: weight_1}
        
        print("\n--- DATASET INFO ---")
        print(f"Normal Count: {count_0} | Scoliosis Count: {count_1}")
        print(f"Calculated Weights: Normal={weight_0:.2f}, Scol={weight_1:.2f}")
        
        # 决定使用哪种 Loss 和 参数
        if USE_FOCAL_LOSS:
            print("\n>>> MODE: Using FOCAL LOSS <<<")
            print(">>> Class weights will NOT be passed to fit() (handled internally by alpha).")
            loss_func = binary_focal_loss(gamma=2.0, alpha=0.25)
            fit_class_weight = None # Focal Loss 不需要外部传入权重
            model_suffix = "_focal"
        else:
            print("\n>>> MODE: Using ORIGINAL BINARY CROSSENTROPY <<<")
            print(">>> Class weights WILL be passed to fit().")
            loss_func = 'binary_crossentropy'
            fit_class_weight = class_weights_dict
            model_suffix = "_original"
            
        print("---------------------\n")

        # ------------------------------------------------------------------
        # B. 构建和编译模型
        # ------------------------------------------------------------------
        model, base_model = build_baseline_model()

        metrics_list = ['accuracy', 
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                        tf.keras.metrics.AUC(name='auc')]

        model.compile(
            optimizer=Adam(learning_rate=INITIAL_LR),
            loss=loss_func, 
            metrics=metrics_list
        )

        # ------------------------------------------------------------------
        # C. 阶段 1: 训练分类头
        # ------------------------------------------------------------------
        checkpoint_phase1 = ModelCheckpoint(
            f'best_scoliosis_model{model_suffix}_phase1.keras', 
            monitor='val_auc',    
            save_best_only=True,    
            mode='max',            
            verbose=1
        )

        print(f"\n=== Phase 1: Training ({'Focal' if USE_FOCAL_LOSS else 'Original'}) ===")
        history = model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=EPOCHS_HEAD,
            class_weight=fit_class_weight, # 动态传入
            callbacks=[checkpoint_phase1] 
        )
        
        # ------------------------------------------------------------------
        # D. 阶段 2: 微调
        # ------------------------------------------------------------------
        print(f"\n=== Phase 2: Fine-Tuning ({'Focal' if USE_FOCAL_LOSS else 'Original'}) ===")
        
        unfreeze_model(base_model, num_layers_to_unfreeze=30)
        
        # 重新编译 (必须保持相同的 loss function)
        model.compile(
            optimizer=Adam(learning_rate=FINE_TUNE_LR),
            loss=loss_func,
            metrics=metrics_list
        )
        
        checkpoint_finetune = ModelCheckpoint(
            f'best_scoliosis_model{model_suffix}_finetuned.keras', 
            monitor='val_auc', 
            save_best_only=True, 
            mode='max', 
            verbose=1
        )
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

        total_epochs = EPOCHS_HEAD + EPOCHS_FINE_TUNE
        
        history_finetune = model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=total_epochs,
            initial_epoch=EPOCHS_HEAD, 
            class_weight=fit_class_weight, # 动态传入
            callbacks=[checkpoint_finetune, reduce_lr, early_stop]
        )
        
        print("\n--- Training Complete ---")
        
        # 加载最佳模型
        best_model_path = f'best_scoliosis_model{model_suffix}_finetuned.keras'
        print(f"Loading best model weights from: {best_model_path}")
        model.load_weights(best_model_path)

        # ------------------------------------------------------------------
        # E. Grad-CAM 生成
        # ------------------------------------------------------------------
        print("\n--- 生成 Grad-CAM 热力图 ---")
        
        valid_gen.reset() 
        val_images, val_labels = next(valid_gen)
        
        scoliosis_indices = np.where(val_labels == 1)[0]
        normal_indices = np.where(val_labels == 0)[0]
        
        selected_indices = []
        selected_indices.extend(scoliosis_indices)
        if len(selected_indices) < 5:
            needed = 5 - len(selected_indices)
            selected_indices.extend(normal_indices[:needed])
        selected_indices = selected_indices[:5]
        
        for i, idx in enumerate(selected_indices):
            print(f"\nProcessing Image {i+1}/5 (Index {idx})...")
            
            img_tensor = val_images[idx].reshape(1, IMG_WIDTH, IMG_HEIGHT, 3)
            true_label_str = "Scoliosis" if val_labels[idx] == 1 else "Normal"
            
            pred_prob = model.predict(img_tensor)[0][0]
            pred_label_str = "Scoliosis" if pred_prob > 0.8 else "Normal"
            
            print(f"  True Label: {true_label_str}, Prediction: {pred_label_str} (Prob: {pred_prob:.4f})")

            heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_layer_name="conv5_block3_out")
            # 文件名带上前缀以便区分
            filename = f"gradcam{model_suffix}_{i+1}_{true_label_str}.jpg"
            save_gradcam_visualization(img_tensor, heatmap, save_path=filename)
        
        print(f"\nGrad-CAM images saved with suffix '{model_suffix}'.")

    except FileNotFoundError:
        print(f"Error: Data directories not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
