import os
import sys
import argparse
from transformers import AutoConfig, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
from models.gpt2_residual import HLMGPT2, HLMGPT2Config
from utils import load_config
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import math

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# 1. 加载数据集
def load_data(prefix):
    logging.info(f'Dataset from {prefix}')
    dataset = load_dataset(os.path.join(prefix, 'htoken.py'),trust_remote_code=True, cache_dir=prefix)
    return dataset

# 2. 配置GPT-2模型和Tokenizer
def load_model(model_name="gpt2", model_config_path='conf/model_config.yaml'):
    gpt2_config = AutoConfig.from_pretrained(model_name)
    vq_config = load_config(model_config_path)
    hlmgpt2_config = HLMGPT2Config(gpt2_config, model_name, vq_config)
    model = HLMGPT2(hlmgpt2_config)
    return model

# 3. 数据预处理：tokenize数据集
def tokenize_data(dataset, tokenizer):
    # we dont need tokenization
    return dataset

# 4. 配置训练参数
def configure_training(model, train_config, train_dataset, val_dataset):
    training_args = TrainingArguments(**train_config)
    #     output_dir="./exp/0222longepoch",          # 保存结果
    #     do_train=True,
    #     do_eval=True,
    #     eval_strategy='epoch',
    #     num_train_epochs=200,              # 训练轮数
    #     per_device_train_batch_size=2,   # 每个设备的训练批次大小
    #     gradient_accumulation_steps=16,   # 梯度累积步数
    #     per_device_eval_batch_size=2,    # 每个设备的评估批次大小
    #     logging_dir="./exp/0222longepoch",            # 日志目录
    #     logging_steps=10,               # 每500步记录日志
    #     save_steps=500,                  # 每500步保存模型
    #     learning_rate=1e-3,               # 学习率
    #     lr_scheduler_type="reduce_lr_on_plateau",        # 学习率调度器类型
    #     max_grad_norm=10,
    #     warmup_steps=1000,               # 预热步数
    #     weight_decay=0.01,              # 权重衰减
    #     adam_beta1=0.9,                      # Adam优化器的beta1参数
    #     adam_beta2=0.95,                 # Adam优化器的beta2参数
    #     ddp_find_unused_parameters=False,
    # )
    
    trainer = Trainer(
        model=model,                        # 要训练的模型
        args=training_args,                 # 训练参数
        train_dataset=train_dataset,   # 训练数据集
        eval_dataset=val_dataset,       # 验证数据集
    )
    return trainer

# 5. 训练模型
def train_model(trainer):
    trainer.train(resume_from_checkpoint=trainer.args.resume_from_checkpoint)
    trainer.evaluate()

# 6. 保存模型
# def save_model(model):
#     model.save_pretrained("./gpt2_finetuned")

# 7. 生成文本
def generate_text(tokenizer):
    generator = pipeline("text-generation", model="./gpt2_finetuned", tokenizer=tokenizer)
    generated_text = generator("This is a test", max_length=50)
    print(generated_text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vq_dir", default=None, help='Path to your vq model and config folder.')
    parser.add_argument("--data_config", default=None, help='Path to your vq data config file. If specified, will cover data_config in `vq_dir`, else will be `vq_dir/data_config.yaml`. At least one of --vq_dir or --data_config is required.')
    parser.add_argument("--model_config", default=None, help='Path to your vq model config file. If specified, will cover model_config in `vq_dir`, else will be `vq_dir/model_config.yaml`. At least one of --vq_dir or --data_config is required.')
    parser.add_argument("--vq_model", default=None, help='Path to your vq model checkpoint. If specified, will cover best_checkpoint in `vq_dir`, else will be `vq_dir/best_checkpoint.pt`. At least one of --vq_dir or --vq_model is required.')
    parser.add_argument("--model_name_or_path", default="/data1/public/hf/openai-community/gpt2", help='Path to gpt2 model')
    parser.add_argument("--train_config", default=None, help="Path to your train config file. If not specified, will be `vq_dir/train_config.yaml`. At least one of --vq_dir or --train_config is required.")
    parser.add_argument("--output_dir", default=None, help='Path to save model and logs. If not specified, will be `vq_dir_hlm`.')
    parser.add_argument("--ckpt_dir", default=None, help='Path to load model for test. If not specified, will use `--output_dir`')
    parser.add_argument("--test", action='store_true', help='Whether to run in test mode.')
    args = parser.parse_args()

    # 0. 处理参数
    if args.vq_dir is None:
        assert args.data_config is not None and args.model_config is not None and args.vq_model is not None and args.output_dir is not None and args.train_config is not None, 'If you dont use --vq_dir, you must specify --data_config, --model_config, --vq_model, --output_dir and --train_config'
    if args.data_config is None:
        args.data_config = os.path.join(args.vq_dir, 'data_config.yaml')
    if args.model_config is None:
        args.model_config = os.path.join(args.vq_dir, 'model_config.yaml')
    if args.train_config is None:
        args.train_config = os.path.join(args.vq_dir, 'train_config.yaml')
    if args.vq_model is None:
        args.vq_model = os.path.join(args.vq_dir, 'best_checkpoint.pt')
    if args.output_dir is None:
        args.output_dir = args.vq_dir
    assert os.path.exists(args.data_config), 'Please check your data_config file path'
    assert os.path.exists(args.model_config), 'Please check your model_config file path'
    # this parameter is not used in this script
    # assert os.path.exists(args.vq_model), 'Please check your vq_model file path'
    assert os.path.exists(args.train_config), 'Please check your train_config file path'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f'Creating output directory {args.output_dir}')
    else:
        logging.warning(f"Output directory {args.output_dir} already exists. May overwrite the existing files.")

    # 1. 加载数据
    logging.info('Loading data...')
    data_config = load_config(config_path=args.data_config)
    data_path = data_config['h5_file_path']
    dataset = load_data(data_path)
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']
    
    # 2. 加载GPT-2模型和Tokenizer
    logging.info('Loading gpt2 model...')
    model_name = args.model_name_or_path
    model = load_model(model_name, model_config_path=args.model_config)

    NUM_QUANTIZER = load_config(args.model_config)['num_quantizers']
    
    # 3. 数据预处理
    logging.info('Tokenizing data')
    # 其实没有什么tokenize的步骤，只不过gpt将它写出来了而已
    # tokenized_datasets = tokenize_data(dataset, tokenizer)
    
    # 4. 配置训练参数
    logging.info('Configuring training...')
    train_config = load_config(config_path=args.train_config)
    if train_config.get('output_dir') is None:
        train_config['output_dir'] = args.output_dir
    if train_config.get('logging_dir') is None:
        train_config['logging_dir'] = train_config['output_dir']
    trainer = configure_training(model, train_config, train_dataset, val_dataset)
    
    if not args.test:
        # 5. 训练模型
        logging.info('Training model...')
        train_model(trainer)
    
        # 6. 保存模型
        logging.info('Saving model...')
        model.save_pretrained(args.output_dir)
    
    # 7. 测试模型
    logging.info('Testing model...')
    model = HLMGPT2.from_pretrained(args.ckpt_dir if args.ckpt_dir is not None else args.output_dir)
    trainer = configure_training(model, train_config, train_dataset, val_dataset)
    eval_metric = trainer.evaluate(test_dataset, metric_key_prefix='test')
    logging.info(f'Test metric: {eval_metric}')
    logging.info(f"Test loss: {eval_metric['test_loss']}")
    logging.info(f"Test loss per codebook: {eval_metric['test_loss']/NUM_QUANTIZER:.4f}")
    logging.info(f"Test ppl: {math.exp(eval_metric['test_loss']/NUM_QUANTIZER):.4f}")

    # 8. 其他测试标准，--test only
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # if args.test:
    if True:
        logging.info("Validation for other metrics...")
        model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            # losses = []
            for batch in tqdm(test_dataset):
                # labels: 1*1024*num_quantizer
                # output: num_quantizer*(1*1024*codebooksize), list
                input_ids = torch.tensor(batch['input_ids']).to(DEVICE)
                input_ids = input_ids.unsqueeze(0)
                labels = torch.tensor(batch['label']).to(DEVICE)
                labels = labels.unsqueeze(0)
                output = model(input_ids=input_ids, labels=labels)
                # loss = output.loss
                # losses.append(loss.item())
                output = output.logits
                output = torch.stack(output).squeeze(dim=1)
                prediction = torch.argmax(output, dim=-1).transpose(0,1).reshape(-1)
                target = labels.view(labels.shape[0], -1)
                target = labels.view(target.shape[0]*NUM_QUANTIZER, -1)
                target = target.view(-1)
                correct += torch.sum(prediction == target).item()
                total += target.shape[0]
        logging.info(f'Codebook prediction accuracy: {correct / total:.4f}')
        logging.info(f"Number of quantizer: {NUM_QUANTIZER}")
        # logging.info(f"Test Loss: {sum(losses) / len(losses)}")

if __name__ == "__main__":
    main()
