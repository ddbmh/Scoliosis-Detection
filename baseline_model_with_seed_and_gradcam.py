import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint # 引入 ModelCheckpoint
import numpy as np
import os
import math 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random # 引入 random

# -------------------------------------------------------------------------
# 新增: 固定随机种子，尽可能保证结果可复现
# -------------------------------------------------------------------------
def reset_random_seeds(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    print(f"Random seeds set to {seed_value}")

reset_random_seeds() # 在程序最开始调用

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
# 修改: 稍微调大学习率，防止 30 epochs 跑不完收敛过程
LEARNING_RATE = 1e-4  
EPOCHS = 30       
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

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x) 
    x = Dense(128, activation='relu')(x) 
    x = Dropout(0.5)(x) 
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def get_data_generators():
    """
    Sets up the training and validation data generators
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,            
        rotation_range=10,         
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
# Grad-CAM 热力图计算
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
    # 再次确保种子设置（防止 main 被外部调用时没设置）
    reset_random_seeds()
    
    model = build_baseline_model()

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )

    model.summary()

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
        
        print("\n--- CLASS WEIGHTS ---")
        print(f"Total Training Samples: {total_samples}")
        print(f"Normal (Class 0) Count: {count_0} -> Weight: {weight_0:.2f}")
        print(f"Scol (Class 1) Count: {count_1} -> Weight: {weight_1:.2f}")
        print("-----------------------\n")
        
        # ------------------------------------------------------------------
        # 新增: ModelCheckpoint 回调函数
        # ------------------------------------------------------------------
        # 这会监控 validation loss，只保存“表现最好”的那一次模型，而不是最后一次
        checkpoint = ModelCheckpoint(
            'best_scoliosis_model.h5', 
            monitor='val_loss',     # 监控验证集 Loss
            save_best_only=True,    # 只保存最好的
            mode='min',             # Loss 越小越好
            verbose=1
        )

        print("\n--- Starting Initial Model Training ---")
        
        history = model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=EPOCHS,
            class_weight=class_weights,
            callbacks=[checkpoint] # 添加回调
        )
        
        print("\n--- Training Complete ---")
        
        # 加载表现最好的模型用于后续的 Grad-CAM 分析
        # 这样即使最后几个 epochs 跑飞了，我们用的也是最好的那个
        print("Loading best model weights for visualization...")
        model.load_weights('best_scoliosis_model.h5')

        # ------------------------------------------------------------------
        # Grad-CAM 生成 (保持不变)
        # ------------------------------------------------------------------
        print("\n--- 生成 5 张 Grad-CAM 热力图 ---")
        
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
        
        print(f"选取验证集图片索引: {selected_indices}")
        
        for i, idx in enumerate(selected_indices):
            print(f"\nProcessing Image {i+1}/5 (Index {idx})...")
            
            img_tensor = val_images[idx].reshape(1, IMG_WIDTH, IMG_HEIGHT, 3)
            true_label_str = "Scoliosis" if val_labels[idx] == 1 else "Normal"
            
            pred_prob = model.predict(img_tensor)[0][0]
            pred_label_str = "Scoliosis" if pred_prob > 0.8 else "Normal"
            print(f"  True Label: {true_label_str}, Prediction: {pred_label_str} (Prob: {pred_prob:.4f})")

            heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_layer_name="conv5_block3_out")
            filename = f"gradcam_result_{i+1}_{true_label_str}.jpg"
            save_gradcam_visualization(img_tensor, heatmap, save_path=filename)
            
        print("\n所有 5 张 Grad-CAM 图片已生成完毕。")

    except FileNotFoundError:
        print(f"Error: Data directories not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
