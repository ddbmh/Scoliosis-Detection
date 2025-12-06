import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import os
import math 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random 

# -------------------------------------------------------------------------
# 1. 固定随机种子 (Reproducibility)
# -------------------------------------------------------------------------
def reset_random_seeds(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    print(f"Random seeds set to {seed_value}")

reset_random_seeds() 

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
# 阶段1学习率 (训练头)
INITIAL_LR = 1e-4  
# 阶段2学习率 (微调，非常小)
FINE_TUNE_LR = 1e-5
EPOCHS_HEAD = 30       
EPOCHS_FINE_TUNE = 20  # 额外的微调轮次

BASE_DATA_DIR = './dataset/' 
TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train')
VALID_DIR = os.path.join(BASE_DATA_DIR, 'validation')


def build_baseline_model(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)):
    """
    Builds the ResNet-50 baseline model for transfer learning.
    """
    base_model = ResNet50(weights='imagenet', 
                          include_top=False, 
                          input_shape=input_shape)

    base_model.trainable = False # 初始阶段冻结

    x = base_model.output
    x = GlobalAveragePooling2D()(x) 
    x = Dense(128, activation='relu')(x) 
    x = Dropout(0.5)(x) 
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model # 返回 base_model 以便后续解冻

def unfreeze_model(base_model, num_layers_to_unfreeze=20):
    """
    解冻 ResNet50 的最后几层进行微调
    """
    base_model.trainable = True
    # 冻结除最后 N 层之外的所有层
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    print(f"\n--- Unfreezing the top {num_layers_to_unfreeze} layers for fine-tuning ---")

def get_data_generators():
    """
    Sets up the training and validation data generators
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,            
        rotation_range=15,         # 稍微增强一点
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

# -------------------------------------------------------------------------
# 2. Grad-CAM 热力图计算函数
# -------------------------------------------------------------------------
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

def main():
    reset_random_seeds()
    
    # 修改：接收 base_model
    model, base_model = build_baseline_model()

    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )

    try:
        train_gen, valid_gen = get_data_generators()
        
        label_0 = train_gen.class_indices['Normal'] 
        label_1 = train_gen.class_indices['Scol'] 

        total_samples = train_gen.samples
        count_0 = np.sum(train_gen.labels == label_0)
        count_1 = np.sum(train_gen.labels == label_1)
        
        weight_0 = (1 / count_0) * (total_samples / 2.0)
        weight_1 = (1 / count_1) * (total_samples / 2.0)

        class_weights = {label_0: weight_0, label_1: weight_1}
        
        # ------------------------------------------------------------------
        # 阶段 1: 训练分类头 (Frozen)
        # ------------------------------------------------------------------
        checkpoint_phase1 = ModelCheckpoint(
            'best_scoliosis_model_phase1.keras', 
            monitor='val_auc',     # 阶段1 监控 AUC
            save_best_only=True,    
            mode='max',            
            verbose=1
        )

        print("\n=== Phase 1: Training Classification Head (Frozen) ===")
        history = model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=EPOCHS_HEAD,
            class_weight=class_weights,
            callbacks=[checkpoint_phase1] 
        )
        
        # ------------------------------------------------------------------
        # 阶段 2: 微调 (Fine-Tuning)
        # ------------------------------------------------------------------
        print("\n=== Phase 2: Fine-Tuning ResNet50 Layers ===")
        
        # 1. 解冻部分层
        unfreeze_model(base_model, num_layers_to_unfreeze=30)
        
        # 2. 重新编译 (必须步骤)，使用极小的学习率
        model.compile(
            optimizer=Adam(learning_rate=FINE_TUNE_LR),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
        )
        
        # 3. 设置微调的回调
        checkpoint_finetune = ModelCheckpoint(
            'best_scoliosis_model_finetuned.keras', 
            monitor='val_auc', 
            save_best_only=True, 
            mode='max', 
            verbose=1
        )
        # 学习率自动衰减：如果 loss 不降，就再减小学习率
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
        # 早停：如果 loss 很久不降，就提前结束
        early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

        # 4. 开始微调训练
        # 注意: initial_epoch 设置为 phase 1 的结束点
        total_epochs = EPOCHS_HEAD + EPOCHS_FINE_TUNE
        
        history_finetune = model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=total_epochs,
            initial_epoch=EPOCHS_HEAD, # 从第 30 轮接着跑
            class_weight=class_weights,
            callbacks=[checkpoint_finetune, reduce_lr, early_stop]
        )
        
        print("\n--- Fine-Tuning Complete ---")
        
        # 加载最好的微调模型进行可视化
        print("Loading best FINE-TUNED model weights for visualization...")
        model.load_weights('best_scoliosis_model_finetuned.keras')

        # ------------------------------------------------------------------
        # Grad-CAM 生成
        # ------------------------------------------------------------------
        print("\n--- 生成 Grad-CAM 热力图 (使用微调后的模型) ---")
        
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
            # 依然用 0.8 阈值演示
            pred_label_str = "Scoliosis" if pred_prob > 0.8 else "Normal"
            
            print(f"  True Label: {true_label_str}, Prediction: {pred_label_str} (Prob: {pred_prob:.4f})")

            heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_layer_name="conv5_block3_out")
            filename = f"gradcam_finetuned_{i+1}_{true_label_str}.jpg"
            save_gradcam_visualization(img_tensor, heatmap, save_path=filename)
            
        print("\n所有图片已生成。对比之前的图，现在应该更聚焦于脊柱。")

    except FileNotFoundError:
        print(f"Error: Data directories not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
