# 多通道波形基实验结果

- 使用轨迹数：30,000
- 训练/测试：22,500 / 7,500
- 重采样点数：64
- NMF 波形基数量：8

## 完整曲线重构
task,split,prefix_ratio,mse,mae,normalized_mse,r2
full_reconstruction,train,1.0,0.0524,0.1197,0.3007,0.1033
full_reconstruction,test,1.0,0.0534,0.1203,0.3043,0.1005

## 前缀预测后续曲线
task,split,prefix_ratio,mse,mae,normalized_mse,r2
future_curve_prediction_unsupervised_nmf,test,0.25,0.1245,0.1803,0.7517,-0.3019
future_curve_prediction_mean_template,test,0.25,0.0754,0.1508,0.4554,-0.0002
future_curve_prediction_prefix_ridge_nmf,test,0.25,0.0748,0.1529,0.452,0.0091
future_curve_prediction_unsupervised_nmf,test,0.5,0.0745,0.1549,0.4747,-0.1392
future_curve_prediction_mean_template,test,0.5,0.0569,0.1429,0.3627,-0.0002
future_curve_prediction_prefix_ridge_nmf,test,0.5,0.0554,0.1422,0.3526,0.0126
future_curve_prediction_unsupervised_nmf,test,0.75,0.053,0.1413,0.345,-0.0474
future_curve_prediction_mean_template,test,0.75,0.0482,0.1415,0.3143,-0.0003
future_curve_prediction_prefix_ridge_nmf,test,0.75,0.0489,0.1405,0.3186,-0.0039

## 未来高峰预测
model,prefix_ratio,channel,channel_zh,positive_rate,majority_accuracy,accuracy,precision,recall,f1
unsupervised_nmf_prefix_fit,0.25,assistant_tokens,模型输出波,0.542,0.542,0.5152,0.5923,0.3387,0.431
prefix_ridge_to_nmf,0.25,assistant_tokens,模型输出波,0.542,0.542,0.462,0.8571,0.0089,0.0175
mean_template,0.25,assistant_tokens,模型输出波,0.542,0.542,0.458,0.0,0.0,0.0
unsupervised_nmf_prefix_fit,0.25,thinking_tokens_est,思考规划波,0.0667,0.9333,0.6853,0.1167,0.566,0.1934
prefix_ridge_to_nmf,0.25,thinking_tokens_est,思考规划波,0.0667,0.9333,0.9305,0.2439,0.02,0.037
mean_template,0.25,thinking_tokens_est,思考规划波,0.0667,0.9333,0.9333,0.0,0.0,0.0
unsupervised_nmf_prefix_fit,0.25,user_tokens_est,环境反馈波,0.1368,0.8632,0.8609,0.1304,0.0029,0.0057
prefix_ridge_to_nmf,0.25,user_tokens_est,环境反馈波,0.1368,0.8632,0.8632,0.0,0.0,0.0
mean_template,0.25,user_tokens_est,环境反馈波,0.1368,0.8632,0.8632,0.0,0.0,0.0
unsupervised_nmf_prefix_fit,0.5,assistant_tokens,模型输出波,0.1679,0.8321,0.7981,0.1028,0.0262,0.0418
prefix_ridge_to_nmf,0.5,assistant_tokens,模型输出波,0.1679,0.8321,0.8276,0.0526,0.0016,0.0031
mean_template,0.5,assistant_tokens,模型输出波,0.1679,0.8321,0.8321,0.0,0.0,0.0
unsupervised_nmf_prefix_fit,0.5,thinking_tokens_est,思考规划波,0.0159,0.9841,0.8777,0.025,0.1765,0.0438
prefix_ridge_to_nmf,0.5,thinking_tokens_est,思考规划波,0.0159,0.9841,0.9841,0.0,0.0,0.0
mean_template,0.5,thinking_tokens_est,思考规划波,0.0159,0.9841,0.9841,0.0,0.0,0.0
unsupervised_nmf_prefix_fit,0.5,user_tokens_est,环境反馈波,0.06,0.94,0.9365,0.0357,0.0022,0.0042
prefix_ridge_to_nmf,0.5,user_tokens_est,环境反馈波,0.06,0.94,0.94,0.0,0.0,0.0
mean_template,0.5,user_tokens_est,环境反馈波,0.06,0.94,0.94,0.0,0.0,0.0
unsupervised_nmf_prefix_fit,0.75,assistant_tokens,模型输出波,0.01,0.99,0.9865,0.0,0.0,0.0
prefix_ridge_to_nmf,0.75,assistant_tokens,模型输出波,0.01,0.99,0.9849,0.0,0.0,0.0
mean_template,0.75,assistant_tokens,模型输出波,0.01,0.99,0.99,0.0,0.0,0.0
unsupervised_nmf_prefix_fit,0.75,thinking_tokens_est,思考规划波,0.0009,0.9991,0.9991,0.0,0.0,0.0
prefix_ridge_to_nmf,0.75,thinking_tokens_est,思考规划波,0.0009,0.9991,0.9991,0.0,0.0,0.0
mean_template,0.75,thinking_tokens_est,思考规划波,0.0009,0.9991,0.9991,0.0,0.0,0.0
unsupervised_nmf_prefix_fit,0.75,user_tokens_est,环境反馈波,0.0228,0.9772,0.9756,0.125,0.0117,0.0214
prefix_ridge_to_nmf,0.75,user_tokens_est,环境反馈波,0.0228,0.9772,0.9769,0.0,0.0,0.0
mean_template,0.75,user_tokens_est,环境反馈波,0.0228,0.9772,0.9772,0.0,0.0,0.0

## 初步解释
- 如果完整曲线重构的测试集归一化均方误差显著小于 1，说明少量非负波形基能解释相当一部分轨迹形态。
- 平均模板是朴素基线；无监督 NMF 前缀拟合检验纯数学模板能否续写；岭回归到 NMF 权重检验轻量机器学习增强是否带来预测提升。
- 如果前缀预测后续曲线的归一化均方误差低于平均模板，说明当前前缀对未来波形有可预测性。
- 如果未来高峰预测的 F1 高于朴素猜测，说明波形基对后续阶段变化有实际预警价值。