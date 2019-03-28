
Training models:

Use `main.py`

```
$ neptune send
```

For sewing,

```
$ neptune send sew.py --config sew.yaml --input ../CIF-50/output/model.pth:model1.pth --input ../CIF-66/output/model.pth:model2.pth
```

---

Uploading data

`neptune data upload --project=brainhackwarsaw/CIFAR10 -r cifar_pytorch`
