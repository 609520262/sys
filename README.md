# 机器学习算法
本项目主要涵盖了机器学习的一些主要算法，目前包括以下几个：

- 监督学习
  * 多元线性回归(Multivariate linear regression)
  * 对率回归(Logistic regression)
  * 线性判别分析(LDA)
  * 支持向量机(SVM)
  * 支持向量描述(SVDD)
  * 决策树(Tree)
- 非监督学习
  * K均值聚类(K-means)
  * 学习向量量化(LVQ)
  * 高斯混合聚类（Mixture-of-Gaussian)
  * 密度聚类(DBSCAN)
- 降维学习
  * MDS
  * PCA
  * 字典学习
****
## 一、模型评估与选择
[CSDN](https://blog.csdn.net/Candy__1/article/details/120642985?spm=1001.2014.3001.5501)
在现实任务中，我们往往有多种学习算法可供选择，甚至对同一个学习算法，当使用不同参数配置时，也会产生不同的模型。我们该选择哪一个学习算法、使用哪一种参数配置呢？这就是机器学习中的**模型选择**问题。模型选择当然是选择出**泛化误差**(模型适应新样本的能力)小的那个模型，那问题又来了，怎么去衡量模型的泛化误差呢？这里我们需要用到一系列的**实验评估方法**获得某种性能度量指标，然后依据指标对学习器性能比较之后得到理想的模型。
 ### **评估方法**
 通常，我们可以通过实验测试来对学习器的泛化误差进行评估。于是，我们需要一个**测试集**或称**验证集**来对模型进行评估。然后在测试集上得到测试误差作为泛化误差的近似。但需注意，测试集应尽可能与训练集互斥，理由是：**考试题目为平时的练习题将失去考试的意义**。
 构造测试集的方法主要有以下几种：
 #### 1.留出法
 留出法直接将数据集划分为两个互斥的集合，一个作为训练集S，一个作为测试集T，即![image](https://user-images.githubusercontent.com/89327936/138209753-c0fe1958-39a5-48ed-abc4-b5e78b78dd35.png).其划分方式及其比例如下图所示：
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/50356263cdc14ef0946df7923bb2bf90.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2FuZHlfXzE=,size_20,color_FFFFFF,t_70,g_se,x_16)
 #### 2.交叉验证法
 交叉验证法先将数据D划分为k个大小相似的互斥子集，即![image](https://user-images.githubusercontent.com/89327936/138209976-80f6e6f9-9f95-4cfb-bc79-ada654b82dbd.png),同样每个子集均来自**分层采样**。每次选择k-1个子集作为训练集，余下的那个子集作为测试集。这样重复以上步骤k次，能够将每个子集都测试一遍，一般，k值常设置为10，称为**10折交叉验证**。
![在这里插入图片描述](https://img-blog.csdnimg.cn/44ace4ac1e6e43adaa8959ff403305cd.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2FuZHlfXzE=,size_20,color_FFFFFF,t_70,g_se,x_16)
  #### 3.自助法
  这种方法借鉴了概率论中的伯努利试验，即采用放回抽样的方法从包含m个样本的数据集D中独立重复进行m次抽取动作得到的样本作为训练集D'，![image](https://user-images.githubusercontent.com/89327936/138210294-e2920978-1f5f-4162-a374-a235b15d651b.png)作为测试集。某个样本在m次采样后始终不被采到的概率为：
 ![image](https://user-images.githubusercontent.com/89327936/138210129-c3f82a4c-6fdd-4097-8a75-0216491b4dda.png)
 ![image](https://user-images.githubusercontent.com/89327936/138210167-c20ecb33-53ea-480b-9855-1965174ac3f3.png)
  即通过自助采样，初始数据集D中约有36.8%的样本未出现在采样数据集D'中
以上三种方法便是常见的实验评估方法，采用上述方法划分训练集、测试集之后，可用训练集训练模型，用测试集测试出相应的性能度量指标。性能度量反应了任务需求，使用不同的性能度量往往会产生不同的评判结果。接下来介绍几种常见的性能度量指标：
   ### **性能度量**
   #### 1.错误率与精度
   用测试集测试模型之后，可得到一系列测试结果，假设1为正例，0为反例：
   错误率是分类错误的样本数占样本总数的比例，精度是分类正确的样本数占样本总数的比例，错误率计算公式：
   
   ![image](https://user-images.githubusercontent.com/89327936/138210468-731b902f-81bf-4edb-a89d-48c202b6d32b.png)
   精度计算公式：
   ![image](https://user-images.githubusercontent.com/89327936/138210493-a02a0a67-2f5e-495f-ba48-b7ad4934db48.png)
   #### 2.P-R曲线及F1度量
   在实际中，错误率与精度并不能满足我们了解整个模型性能的全部需求，例如我们想知道有多少样本是反例而被误判成了正例或者有多少样本是正例而被误判成了反例，显然错误率和精度不能体现上述指标。
   对于二分类问题，我们常常用混淆矩阵来度量我们所需要的结果：
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/210c180afcb641f3ad286753cd2f3736.png)
   
   查准率P与查全率R可以这样定义：
   ![image](https://user-images.githubusercontent.com/89327936/138210615-e7d020b7-c7ee-418a-871d-204901e03212.png)
   ![image](https://user-images.githubusercontent.com/89327936/138210640-4eb8e72f-3301-4885-b252-76f721462983.png)
  
   对于P和R我们可以这样理解，查准率P就是在所测试的样本中的准确程度，准确度越高越好，即分母应该是预测为正例的样本和；相反，查全率希望真实样本中的正例被尽可能多的调出来，越全越好，即分母应该是真实值为正例的样本和。
   查准率和查全率往往是一对矛盾量，若想提高查全率则势必会降低查准率，我们期待的是查准率和查全率均得到较高的值，一种度量手段便是绘制**P-R曲线**。
   P-R曲线的生成方法：根据学习器的预测结果对样本进行排序，排在前面的是学习器认为最可能是正例的样本，排在最后的是最不可能是正例的样本，按此顺序逐个将样本作为正例预测，则每次可以计算出当前的查全率、查准率，以查全率为横轴、查准率为纵轴做图，得到的查准率-查全率曲线即为P-R曲线。
   
   ![image](https://user-images.githubusercontent.com/89327936/138210669-353bf782-6cf3-4926-a1f0-7e07beea8edd.png)
P-R曲线直观地显示了学习器在样本总体上的查全率、查准率。在比较时，若一个学习器的P-R曲线被另一个学习器完全包住，则可断言后者的性能由于前者。
   如上图中能够看出学习器B性能优于A，学习器性能C优于A，学习器性能D优于A,但学习器B、C、D该如何比较？
   我们采用**平衡点**(Break-Event Point)进行度量，它是**查准率=查全率**时的取值，
![在这里插入图片描述](https://img-blog.csdnimg.cn/7d111396c5454374b2d89716fd5b8cca.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2FuZHlfXzE=,size_20,color_FFFFFF,t_70,g_se,x_16)
 同样，平衡点距离原点越远，性能越好，通过比较平衡点可得出学习器B=C性能由于D，但学习器B和C的性能就真是就相等了么？，显然是不一样的。
   可见光从P-R曲线也不太能反应出学习器的全部信息，我们可利用F1度量方式对B和C的性能进行度量。
   F1度量时基于查准率和查全率的调和平均定义的：
  ![image](https://user-images.githubusercontent.com/89327936/138210893-37a0fe8d-cc4e-4c5e-a550-204bbea3401b.png)
  
   #### 3.ROC与AUC
   ROC全称“受试者工作特征”(Receiver Operating Characteristic)曲线。类比于P-R曲线，我们这里更换了P-R曲线的横纵坐标，其中纵坐标为TPR(真正例率)，横坐标为FPR(假正例率）：
   ![image](https://user-images.githubusercontent.com/89327936/138210912-efa31f61-2b38-4e3e-b2c5-f1ddf33c09cc.png)
   ![image](https://user-images.githubusercontent.com/89327936/138210937-41391051-28c9-4906-955b-03175a2726cd.png)
   细心朋友会发现，真正例率TPR其实就是前面所说的查全率R，这里只是换了一个名字。
   有混淆举证可知，我们希望真正例率TPR尽可能大，而假正例率FPR尽可能小，于是在ROC图中我们希望曲线能位于左上部分。
   
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/d99161c74ecc42d9a24b0be25a22bd17.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2FuZHlfXzE=,size_20,color_FFFFFF,t_70,g_se,x_16)
    进行学习器比较时，与P-R曲线相似，若一个学习器的ROC曲线被另一个学习器曲线完全抱住，则可断言后者性能优于前者；若两个学习器的ROC曲线发生交叉，则难以一般性地断言孰优孰劣。此时较为合理的判据是ROC面积，即**AUC**。
   ![image](https://user-images.githubusercontent.com/89327936/138211063-3fa20ae0-2cf3-4c4c-983a-cb4fac932eba.png)
   #### 4.代价曲线
 在现实任务中，我们常常会遇到这种情况：例如在医疗诊断中错误吧患者诊断为健康人与错误地把健康人诊断为患者，看起来都是犯了“一次错误”，但后者的影响是增加了进一步诊断的麻烦，前者的结果却可能是丧失了拯救生命的最佳时机。为权衡不同类型错误所造成的不同损失，可为不同的错误定义相应的权重，称为“非均等代价”。
 例如，我们可定义如下代价矩阵：
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/24cc2c5fe1ea455ab8907bb385c5a257.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2FuZHlfXzE=,size_20,color_FFFFFF,t_70,g_se,x_16)
  于是，可修改前面的精度公式：
   ![image](https://user-images.githubusercontent.com/89327936/138211182-b2600ab5-740b-4ec6-ab3f-273bf91b70bd.png)
  在非均等代价下，我们的绘制代价曲线图：
  横轴：![在这里插入图片描述](https://img-blog.csdnimg.cn/e8f4cdaa56ed4488bdb844bfa66f4b97.png)
  纵轴：![在这里插入图片描述](https://img-blog.csdnimg.cn/f5e3313642b2444dab64de1694bf9a89.png)
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/28a17941eaba43b0bdf933e5821d7a5a.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2FuZHlfXzE=,size_20,color_FFFFFF,t_70,g_se,x_16)
