# tokyo_photo2geo

Predict where the photo was taken from single shot image

![](./doc/result.gif)

# Predict your photo

TBA

# Create your own model

```python
poetry install
KEY=XXXXXX poetry run python scripts/create_dataset.py
poetry run python scripts/train.py \
    --model resnet50 \
    --batch_size 30 \
    --lr 0.00001 \
    --weight_decay 0 \
    --epoch 200
```

# References

TBA
