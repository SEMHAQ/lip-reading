import tensorflow as tf
import os
import argparse
import pandas as pd
from utils import three_d_resnet_bi_lstm, datagenerator, sequence_modeling
from config import EMO_MAP

# 硬件配置
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main(args):
    # 1. 基础超参数
    len_seq = 60
    size = 88
    chan = 3
    bs = 8
    n_cl = 12  # 你的唇读短语分类数
    INIT_LR = 0.0001
    EPOCHS = 20  # 建议20次以保证收敛效果

    # 2. 加载数据索引 (假设你已经生成了包含 path, emotion, speaker 的 csv)
    # 你需要准备好 iemocap_metadata.csv 和 meld_metadata.csv
    df_iemocap = pd.read_csv('metadata/iemocap_metadata.csv')
    df_meld = pd.read_csv('metadata/meld_metadata.csv')
    df_all = pd.concat([df_iemocap, df_meld])

    # 3. 根据策略筛选情感标签
    target_emotions = EMO_MAP[args.strategy][args.category]
    print(f"--- 正在训练专家模型: {args.strategy}_{args.category} ---")
    print(f"--- 目标情感标签: {target_emotions} ---")

    # 4. 数据拆分 (Speaker-Independent)
    # IEMOCAP通常用Session 5做测试，MELD有官方划分
    df_train = df_all[df_all['split'] == 'train']
    df_val = df_all[df_all['split'] == 'val']

    # 核心：过滤出该专家模型负责的情感数据
    df_train_expert = df_train[df_train['emotion'].isin(target_emotions)]
    df_val_expert = df_val[df_val['emotion'].isin(target_emotions)]

    # 5. 生成器准备
    # 使用你原来的 sequence_modeling 逻辑处理路径
    p_train, _, _, l_train = sequence_modeling.get_sequence(df_train_expert, len_seq=len_seq)
    p_val, n_val, _, l_val = sequence_modeling.get_sequence(df_val_expert, len_seq=len_seq)

    train_gen = datagenerator.DataGeneratorTrain(p_train, l_train, shape=(size, size), bs=bs)
    valid_gen = datagenerator.DataGeneratorTest(p_val, l_val, n_val, shape=(size, size), bs=bs)

    # 6. 构建模型 (3D-ResNet18 + Bi-LSTM)
    model_name = f"LipExpert_{args.strategy}_{args.category}"
    model = three_d_resnet_bi_lstm.build_three_d_resnet_18(
        (len_seq, size, size, chan), n_cl, 'softmax', None, True, model_name
    )

    # 7. 编译与训练
    opt = tf.keras.optimizers.Adam(learning_rate=INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # 权重保存路径
    checkpoint_path = f"weights/lip/{args.strategy}/{args.category}/"
    if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path)

    # 自定义Callback记录结果（对应你代码里的metrics）
    # 这里简化为自带的 ModelCheckpoint
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path + "best_model.h5",
        save_best_only=True,
        monitor='val_accuracy'
    )

    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS,
        callbacks=[cp_callback],
        verbose=1
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, required=True, help='AW, PN, or AC')
    parser.add_argument('--category', type=str, required=True, help='e.g., high, Type_B, Approach')
    args = parser.parse_args()
    main(args)
