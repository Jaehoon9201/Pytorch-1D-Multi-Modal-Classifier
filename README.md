# Python-1D-Multi-Modal-Classifier
Python-1D-Multi-Modal-Classifier for simple data

## Modeling for Multi-Modal structure
First layer is splitted for Multi-Modal structure.
```python
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.l11 = nn.Linear(5, 16)
        self.l12 = nn.Linear(5, 16)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 7)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x11 = x[:, 0:5]
        x12 = x[:, 5:10]
        x11 = self.relu(self.l11(x11))
        x12 = self.relu(self.l12(x12))

        x11 = x11.view(-1, 16)
        x12 = x12.view(-1, 16)
        x = torch.cat([x11, x12], dim=1)
        x = self.relu(self.l2(x))
        return self.l3(x)

```
## Results of this code
![image](https://user-images.githubusercontent.com/71545160/130219570-5946436f-9073-473b-8763-48ee0f8bc87b.png)

![image](https://user-images.githubusercontent.com/71545160/130219533-3bab7f52-a34f-4149-9efc-120d47c8a9ee.png)


