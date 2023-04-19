# WL_loss easiest way to use multiple loss functions

# Usage
```python
loss_fns = [loss_function1, loss_function2, loss_function3]
loss_fns_weight = [loss1_weight, loss2_weight, loss3_weight]
loss = WL_loss(loss_fns=loss_fns, loss_fns_weight=loss_fns_weight)
```
then you can use object loss as a normal keras loss function

# Loading
this class in complitly serialized so you can load your model with ease just pass this class as a custom object
```python
model = tf.keras.models.load_model(model_name, custom_objects={"WS_loss": WS_loss})
```
