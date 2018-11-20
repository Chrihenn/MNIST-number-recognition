import mxnet as mx
import logging

# henter ut mnist dataset
mnist = mx.test_utils.get_mnist()

#Seed
mx.random.seed(42)

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

batch_size = 100
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

data = mx.sym.var('data')
data = mx.sym.flatten(data=data)

fc1 = mx.sym.FullyConnected(data=data, num_hidden=128)
act1 = mx.sym.Activation(data=fc1, act_type="relu")

fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64)
act2 = mx.sym.Activation(data=fc2, act_type="relu")

logging.getLogger().setLevel(logging.DEBUG)

fc3 = mx.sym.FullyConnected(data=act2, num_hidden=10)
mlp = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

mlp_model = mx.mod.Module(symbol=mlp, context=ctx)
mlp_model.fit(train_iter, eval_data=val_iter, optimizer='sgd', optimizer_params={'learning_rate': 0.1},
              eval_metric='acc', batch_end_callback=mx.callback.Speedometer(batch_size, 100),
              num_epoch=10)

test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
prob = mlp_model.predict(test_iter)
assert prob.shape == (10000, 10)

test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

acc = mx.metric.Accuracy()
mlp_model.score(test_iter, acc)
print(acc)                                    # Her har programemt en accuracy på 96%
assert acc.get()[1] > 0.96, "Achieved accuracy (%f) is lower than expected (0.96)" % acc.get()[1]

data = mx.sym.var('data')

conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))

conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

lenet_model = mx.mod.Module(symbol=lenet, context=ctx)

lenet_model.fit(train_iter, eval_data=val_iter, optimizer='sgd', optimizer_params={'learning_rate': 0.1},
                eval_metric='acc', batch_end_callback= mx.callback.Speedometer(batch_size, 100),
                num_epoch=10)

test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
prob = lenet_model.predict(test_iter)
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

acc = mx.metric.Accuracy()
lenet_model.score(test_iter, acc)
print(acc)                                     # Her går accuracyen opp med 2%
assert acc.get()[1] > 0.98, "Achieved accuracy (%f) is lower than expected (0.98)" % acc.get()[1]
