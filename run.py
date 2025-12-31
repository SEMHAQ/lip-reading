import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc
import h5py
import json
import warnings

warnings.filterwarnings('ignore')


def initialize_environment():
    """初始化运行环境"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')

    # 设置随机种子
    np.random.seed(42)
    tf.random.set_seed(42)

    # 检查GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    return len(gpus) > 0


def load_trained_model(model_path):
    """加载预训练的H5模型"""
    try:
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")

        # 加载模型
        model = load_model(model_path, compile=False)

        # 编译模型
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        return model
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        sys.exit(1)


def preprocess_test_data(data_dir, input_shape):
    """预处理测试数据"""
    # 这里应该有实际的数据加载逻辑
    # 为了示例，我们生成随机数据
    num_samples = 1000
    num_classes = 10

    # 生成随机测试数据
    X_test = np.random.randn(num_samples, *input_shape).astype('float32')

    # 生成随机标签
    y_true = np.random.randint(0, num_classes, num_samples)
    y_pred_proba = np.random.rand(num_samples, num_classes)

    # 归一化预测概率
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)

    # 获取预测类别
    y_pred = np.argmax(y_pred_proba, axis=1)

    return X_test, y_true, y_pred, y_pred_proba


def calculate_uac_score(y_true, y_pred_proba):
    """计算UAC（Unified Accuracy Curve）分数"""
    n_classes = y_pred_proba.shape[1]
    uac_scores = []

    for class_idx in range(n_classes):
        # 获取当前类别的真实标签和预测概率
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = y_pred_proba[:, class_idx]

        # 计算精确率-召回率曲线下的面积
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_binary)
        uac_scores.append(auc(recall, precision))

    # 返回平均UAC
    return np.mean(uac_scores)


def calculate_f1_score_sklearn(y_true, y_pred):
    """计算F1分数"""
    # 计算宏平均F1
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # 计算微平均F1
    f1_micro = f1_score(y_true, y_pred, average='micro')

    return f1_macro, f1_micro


def evaluate_model_performance(model, X_test, y_true):
    """评估模型性能"""
    # 获取模型预测
    y_pred_proba = model.predict(X_test, batch_size=32, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # 计算准确率
    acc = accuracy_score(y_true, y_pred)

    # 计算UAC
    uac = calculate_uac_score(y_true, y_pred_proba)

    # 计算F1分数
    f1_macro, f1_micro = calculate_f1_score_sklearn(y_true, y_pred)

    # 计算精确率和召回率
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    # 计算每个类别的性能指标
    class_metrics = []
    n_classes = len(np.unique(y_true))

    for i in range(n_classes):
        class_precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        class_recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (
                                                                                                        class_precision + class_recall) > 0 else 0

        class_metrics.append({
            'class': i,
            'precision': class_precision,
            'recall': class_recall,
            'f1': class_f1
        })

    return {
        'accuracy': acc,
        'uac': uac,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'class_metrics': class_metrics
    }


def generate_classification_report(metrics):
    """生成分类报告"""
    report = {
        'overall': {
            'accuracy': float(metrics['accuracy']),
            'uac': float(metrics['uac']),
            'f1_macro': float(metrics['f1_macro']),
            'f1_micro': float(metrics['f1_micro']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall'])
        },
        'per_class': metrics['class_metrics'],
        'timestamp': tf.timestamp().numpy().item()
    }

    return report


def save_evaluation_results(results, output_dir):
    """保存评估结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存JSON结果
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # 保存混淆矩阵
    np.save(os.path.join(output_dir, 'confusion_matrix.npy'), results['confusion_matrix'])

    # 保存文本报告
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("模型评估报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("整体性能指标:\n")
        f.write("-" * 40 + "\n")
        for key, value in results['overall'].items():
            f.write(f"{key:15}: {value:.4f}\n")

        f.write("\n各类别性能指标:\n")
        f.write("-" * 40 + "\n")
        for metric in results['per_class']:
            f.write(f"类别 {metric['class']}:\n")
            f.write(f"  精确率: {metric['precision']:.4f}\n")
            f.write(f"  召回率: {metric['recall']:.4f}\n")
            f.write(f"  F1分数: {metric['f1']:.4f}\n")

    return report_path


def load_config(config_path):
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # 默认配置
        config = {
            'model_path': 'models/trained_model.h5',
            'input_shape': [224, 224, 3],
            'num_classes': 10,
            'batch_size': 32,
            'test_data_dir': 'data/test',
            'output_dir': 'results',
            'random_seed': 42
        }

    return config


def validate_model_architecture(model):
    """验证模型架构"""
    # 检查模型层
    layers = model.layers
    layer_types = [layer.__class__.__name__ for layer in layers]

    # 检查输出形状
    input_shape = model.input_shape
    output_shape = model.output_shape

    # 检查参数数量
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

    return {
        'num_layers': len(layers),
        'layer_types': layer_types,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'total_params': trainable_params + non_trainable_params
    }


def create_dummy_data(input_shape, num_samples):
    """创建虚拟数据用于测试"""
    # 创建随机输入数据
    X = tf.random.normal([num_samples] + list(input_shape))

    # 创建随机标签
    y = tf.random.uniform([num_samples], maxval=10, dtype=tf.int32)

    # 转换为one-hot编码
    y_one_hot = tf.one_hot(y, depth=10)

    return X, y_one_hot, y


def main():
    """主函数"""
    # 初始化环境
    use_gpu = initialize_environment()
    print(f"GPU可用: {use_gpu}")

    # 加载配置
    config = load_config('config.json')

    # 加载模型
    model = load_trained_model(config['model_path'])

    # 验证模型架构
    model_info = validate_model_architecture(model)
    print(f"模型参数总数: {model_info['total_params']:,}")

    # 创建测试数据
    X_test, y_test_one_hot, y_test = create_dummy_data(
        tuple(config['input_shape']),
        1000
    )

    # 评估模型
    metrics = evaluate_model_performance(model, X_test, y_test.numpy())

    # 生成报告
    report = generate_classification_report(metrics)

    # 保存结果
    output_path = save_evaluation_results(report, config['output_dir'])

    # 输出关键指标
    print("\n" + "=" * 60)
    print("关键性能指标:")
    print("=" * 60)
    print(f"Accuracy (ACC): {metrics['accuracy']:.4f}")
    print(f"UAC Score:      {metrics['uac']:.4f}")
    print(f"F1 Macro:       {metrics['f1_macro']:.4f}")
    print(f"F1 Micro:       {metrics['f1_micro']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print("=" * 60)

    # 保存模型性能快照
    snapshot = {
        'model_info': model_info,
        'metrics': metrics,
        'config': config,
        'timestamp': tf.timestamp().numpy().item()
    }

    snapshot_path = os.path.join(config['output_dir'], 'model_snapshot.json')
    with open(snapshot_path, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)

    print(f"\n评估结果已保存至: {output_path}")
    print(f"模型快照已保存至: {snapshot_path}")

    return metrics


if __name__ == "__main__":
    # 运行主程序
    results = main()

    # 程序退出代码
    sys.exit(0)