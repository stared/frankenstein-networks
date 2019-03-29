
Training models:

Use `main.py`

```
$ neptune send
```

For sewing

```
$ neptune send --config sew.yaml --input cifar_pytorch --input /CIF-52/output/model.pth:model1.pth --input /CIF-52/output/model.pth:model2.pth
```

---

Uploading data

`neptune data upload --project=brainhackwarsaw/CIFAR10 -r cifar_pytorch`
