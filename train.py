from models import model
import tensorflow as tf
from tensorflow.keras import metrics,losses,optimizers
import settings
from data import make_dataset
if __name__ == '__main__':
    dataset = make_dataset(settings.train)
    dataset_test = make_dataset(settings.test)
    loss_object = losses.categorical_crossentropy
    acc_meter = metrics.CategoricalAccuracy()
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss=loss_object,
        metrics=['accuracy']
    )
    for epoch in range(1, settings.epoch):
        model.fit(dataset, batch_size=settings.batch_size)
        if epoch%settings.each_epoch==0:
            if settings.out_model:
                tf.saved_model.save(model, settings.out_model+'/model-savedmodel')
            print('saving savedmodel.')
            for x1, y1 in dataset_test:  # 遍历测试集
                pred = model(x1)  # 前向计算
                acc_meter.update_state(y_true=y1, y_pred=pred)  # 更新准确率统计
            print("测试集正确率为：", acc_meter.result())
            acc_meter.reset_states()