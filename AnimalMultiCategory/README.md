## 动物多分类项目说明文档
此项目是进行动物多分类实战，包括“纲”分类（哺乳动物或者鸟类）、“种”分类(是兔子、老鼠还是鸡)、同时预测“纲”和种

### 一. 各文件夹及文件说明
#### 1. 文件夹说明
ByTensorflow: 利用Tensorflow框架进行网络模型训练;

Datas: 图片数据(训练数据、验证数据和测试数据); 1735个样本，其中rabbits 670个样本，rats 516个样本，chickens 542个样本。 验证集有 163个样本， 
rabbits 60个样本, rats 51个样本, chickens 52个样本;

Datasets: 对图片数据处理的相关工具类;

Stage_1 Classes_classification: 完成动物纲(Classes)分类，预测该动物是属于哺乳纲(Mammals)还是鸟纲(Birds);

Stage_2 Species_classification: 成动物种(Species)分类，预测该动物是兔子、老鼠还是鸡;

Stage_3 Multi_classification: 完成多任务分类,同时预测该动物的“纲”和“种”;

models: 保存训练好的模型

#### 2. 文件说明
Image_rename.py: 对Datas中的train和val文件夹中的文件夹进行统一命名操作。 这个是整个项目首先需要运行的文件，切记！！！！

Image_covert_data.py: 此文件是对图片信息进行二进制保存。每张图片以{'path':'','classes':'','species':''}进行二进制存储，方便后面直接
读取此二进制文件即可。生成的文件分别存储在Datas目录下的train_annotation.npy 和val_annotation.npy中。
这个文件需要在Image_rename.py后运行。一定要运行，否则后面的三种分类情况的数据无法获得。

Config.py: 配置文件，包含模型训练图片的长宽、字符串类别对应int型label的字典

Helper.py: 帮助类，one-hot 编码、生成批次迭代器、日志写入、图片和预测结果显示

下面说明每个模型中相关文件:

a. Stage_1 Classes_classification

Classes_make_anno.py : 切忌！！！！ ,如果本地没有转换后特征和目标属性数据，这个文件一定要最先运行！！！！  功能是读取保存的二进制文件，转换为Classes_train_features.npy、Classes_train_label.npy、
Classes_val_features.npy、Classes_val_label.npy。 

Classes_Network.py : 用于“纲”分类的网络。

Classes_classification.py : 程序主体，用来进行模型训练/验证。

Classes_test.py ： 用测试图片集进行预测，可视化查看结果。


b. Stage_2 Species_classification

Species_make_anno.py : 切忌！！！！ ,如果本地没有转换后特征和目标属性数据，这个文件一定要最先运行！！！！  功能是读取保存的二进制文件，转换为Species_train_features.npy、Species_train_label.npy、
Species_val_features.npy、Species_val_label.npy。 

Species_Network.py : 用于“种”分类的网络。

Species_classification.py : 程序主体，用来进行模型训练/验证。

Species_test.py ： 用测试图片集进行预测，可视化查看结果。

c. Stage_3 Multi_classification

Multi_make_anno.py : 切忌！！！！ ,如果本地没有转换后特征和目标属性数据，这个文件一定要最先运行！！！！  功能是读取保存的二进制文件，转换为Multi_train_features.npy
Multi_train_labels_classes.npy、Multi_train_labels_species.npy、Multi_val_features.npy、Multi_val_features.npy、Multi_val_labels_species.npy。 

Multi_Network.py : 用于“种”和“纲”分类的网络。

Multi_classification.py : 程序主体，用来进行模型训练/验证。

Multi_test.py ： 用测试图片集进行预测，可视化查看结果。

#### 3. 动物纲(Classes)分类 模型详细说明
a. 运行此模型，需要首先运行Classes_make_anno.py，生成特征数据和目标数据。 这里取了全部的数据，并进行了数据打乱，讲老鼠和兔子归为哺乳纲，鸡归为
鸟纲，以*.npy的形式存储到本地，模型训练时直接加载.npy文件。

b. 运用了ResNet 50 网络结构。 输入图片数据大小为[224, 224, 3], 经过一系列卷积操作后得到一个1*2048的向量，对其进行全连接操作，得到2个输出值
，进行softmax操作，得到相应的概率。

![avatar](https://img-blog.csdn.net/20170405213708608?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvU2V2ZW5feWVhcl9Qcm9taXNl/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

![avatar](https://img-blog.csdn.net/20170405213726978?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvU2V2ZW5feWVhcl9Qcm9taXNl/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

![avatar](https://img-blog.csdn.net/20170405213826979?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvU2V2ZW5feWVhcl9Qcm9taXNl/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

![avatar](https://img-blog.csdn.net/20170405213946417?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvU2V2ZW5feWVhcl9Qcm9taXNl/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

#### 4. 动物种(Species)分类，预测该动物是兔子、老鼠还是鸡 模型跟3 差不多，这里不做详细说明
运行此模型，需要首先运行Species_make_anno.py，生成特征数据和目标数据。

#### 5. 完成多任务分类,同时预测该动物的“纲”和“种” 
运行此模型，需要首先运行Multi_make_anno.py，生成特征数据和目标数据。
    
这里是多标签分类，前面的卷次层和fc1和3，4是一致的。 最后的输出层调整了下，分别获取两种标签的输出值，并计算这两种的loss值，将两种loss值相加，进行模型训练。
    






