# Image Captioning Baseline
본 소스 코드는 '국립국어원 인공 지능 언어 능력 평가' 시범 운영 과제 중 '문자가 포함된 이미지 기반 문장 생성' 과제 베이스라인 모델 및 학습과 평가를 위한 코드입니다.

학습 및 추론, 평가는 아래의 실행 방법(How to Run)에서 확인하실 수 있습니다.  

|Model|ROUGE-1|BLUE|
|:---:|---|---|
|ViT + KoGPT2|0.3191|0.4067|

## 디렉토리 구조(Directory Structure)
```
# 학습에 필요한 리소스들이 들어있습니다.
resource
├── data
└── tokenizer

# 실행 가능한 python 스크립트가 들어있습니다.
run
├── infernece.py
└── train.py

# 학습에 사용될 커스텀 함수들이 구현되어 있습니다.
src
├── data.py     # torch dataloader
├── module.py   # pytorch-lightning module
└── utils.py
```

## 데이터(Data)
### 제공 데이터
```
{
    "id": "nikluge-gips-2023-train-000000",
    "input": {
        "id": "P00001",
        "image_width": 6000,
        "image_height": 4000,
        "ocr_info": [
            {
                "words": "2인승",
                "type": "rect",
                "bbox": {
                    "x": 486,
                    "y": 1091,
                    "width": 891,
                    "height": 193
                }
            }
        ]
    },
    "output": [
        "2인승이라고 적혀 있는 하늘색 테두리 안내판 옆에는 다양한 색깔의 우산들이 띄워져 있다.",
        "다양한 색깔의 우산들이 띄워져 있는 곳 옆에는 2인승이라고 쓰인 하늘색 테두리 안내판이 있다,",
        "다양한 색깔의 우산이 띄워진 곳 옆에 있는 하늘색 테두리 안내판에는 2인승이라고 적혀 있다.",
        "하늘색 테두리 안내판에는 2인승이라고 적혀 있는데, 그 옆에 띄워져 있는 것은 다양한 색깔의 우산들이다.",
        "안내판은 하늘색 테두리에 2인승이라고 적혀 있으며, 그 옆에 띄워져 있는 것은 다양한 색의 우산들이다."
    ]
}
...
```
`input`은 이미지 파일명입니다.

## 설치(Installation)
Execute it, if mecab is not installed
```
./install_mecab.sh
```

Install python dependency
```
pip install -r requirements.txt
```

## 실행 방법(How to Run)
### 학습(Train)
```
python -m run train \
    --output-dir outputs/ic \
    --seed 42 --epoch 10 --gpus 4 --warmup-rate 0.1 \
    --max-learning-rate 2e-4 --min-learning-rate 1e-5 \
    --batch-size=32 --valid-batch-size=64 \
    --logging-interval 100 --evaluate-interval 1 \
    --wandb-project <wandb-project-name>
```
- 기본 모델은 `google/vit-base-patch16-224-in21k`와 `skt/kogpt2-base-v2`를 이용합니다.
- 학습 로그 및 모델은 지정한 `output-dir`에 저장됩니다.

### 추론(Inference)
```
python -m run inference \
    --model-ckpt-path outputs/ic/<your-model-ckpt-path> \
    --output-path test_output.jsonl \
    --batch-size=64 \
    --output-max-seq-len 512 \
    --num-beams 5 \
    --device cuda:0
```
- `transformers` 모델을 불러와 inference를 진행합니다.
- Inference 시 출력 데이터는 jsonl format으로 저장되며, "output"의 경우 입력 데이터와 다르게 `list`가 아닌 `string`이 됩니다.

### 채점(scoring)
```
python -m run scoring \
    --candidate-path <your-candidate-file-path>
```
- Inference output을 이용해 채점을 진행합니다.
- 기본적으로 Rouge-1과 BLEU를 제공합니다.

## Reference

huggingface/transformers (https://github.com/huggingface/transformers)  
SKT-AI/KoGPT2 (https://github.com/SKT-AI/KoGPT2)  
국립국어원 모두의말뭉치 (https://corpus.korean.go.kr/)  
