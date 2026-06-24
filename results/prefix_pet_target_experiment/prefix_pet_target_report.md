# 前缀续写与机器学习增强实验

- 轨迹数：30,000
- 前缀样本数：90,000
- 前缀比例：0.25, 0.5, 0.75
- 波形基数量：8
- 监督模型：sklearn HistGradientBoosting fallback

## 第三步：前缀续写指标
prefix_ratio,model,channel,channel_zh,future_curve_normalized_mse_mean,future_peak_positive_rate,future_peak_accuracy,future_peak_balanced_accuracy,future_peak_f1
0.25,mean_template,assistant_tokens,模型输出波,0.4572,0.5336,0.4664,0.5,0.0
0.25,unsupervised_nmf_prefix_fit,assistant_tokens,模型输出波,0.5263,0.5336,0.476,0.5027,0.177
0.25,mean_template,thinking_tokens_est,思考规划波,0.4572,0.0679,0.9321,0.5,0.0
0.25,unsupervised_nmf_prefix_fit,thinking_tokens_est,思考规划波,0.5263,0.0679,0.8287,0.6119,0.2226
0.25,mean_template,user_tokens_est,环境反馈波,0.4572,0.1369,0.8631,0.5,0.0
0.25,unsupervised_nmf_prefix_fit,user_tokens_est,环境反馈波,0.5263,0.1369,0.863,0.5001,0.001
0.5,mean_template,assistant_tokens,模型输出波,0.4235,0.1685,0.8315,0.5,0.0
0.5,unsupervised_nmf_prefix_fit,assistant_tokens,模型输出波,0.5008,0.1685,0.7809,0.5666,0.2723
0.5,mean_template,thinking_tokens_est,思考规划波,0.4235,0.0179,0.9821,0.5,0.0
0.5,unsupervised_nmf_prefix_fit,thinking_tokens_est,思考规划波,0.5008,0.0179,0.9474,0.4997,0.0235
0.5,mean_template,user_tokens_est,环境反馈波,0.4235,0.059,0.941,0.5,0.0
0.5,unsupervised_nmf_prefix_fit,user_tokens_est,环境反馈波,0.5008,0.059,0.8132,0.56,0.147
0.75,mean_template,assistant_tokens,模型输出波,0.3197,0.0118,0.9882,0.5,0.0
0.75,unsupervised_nmf_prefix_fit,assistant_tokens,模型输出波,0.3312,0.0118,0.9847,0.4982,0.0
0.75,mean_template,thinking_tokens_est,思考规划波,0.3197,0.0018,0.9982,0.5,0.0
0.75,unsupervised_nmf_prefix_fit,thinking_tokens_est,思考规划波,0.3312,0.0018,0.9982,0.5,0.0
0.75,mean_template,user_tokens_est,环境反馈波,0.3197,0.0229,0.9771,0.5,0.0
0.75,unsupervised_nmf_prefix_fit,user_tokens_est,环境反馈波,0.3312,0.0229,0.977,0.5007,0.0029

## 第四步：机器学习增强指标
feature_group,target,model_family,features,positive_rate,accuracy,balanced_accuracy,f1,auc,macro_f1,weighted_f1,mae_turns,median_abs_error_turns,r2_log_remaining
stats_only,future_end_within_5,sklearn_hist_gradient_boosting,51,0.0004,0.9997,0.8333,0.6667,0.7778,,,,,
stats_only,future_long_output,sklearn_hist_gradient_boosting,51,0.5144,0.7053,0.7028,0.7333,0.7789,,,,,
stats_only,future_overheat,sklearn_hist_gradient_boosting,51,0.5905,0.6439,0.6046,0.7316,0.6815,,,,,
stats_only,future_peak_any,sklearn_hist_gradient_boosting,51,0.2843,0.8252,0.7788,0.6858,0.8938,,,,,
stats_only,pet_state_multiclass,sklearn_hist_gradient_boosting,51,,0.9996,0.9992,,,0.999,0.9996,,,
stats_only,remaining_turns_research,sklearn_hist_gradient_boosting,51,,,,,,,,0.5634,0.4934,0.9977
waveform_only,future_end_within_5,sklearn_hist_gradient_boosting,22,0.0004,0.9996,0.7777,0.5263,0.8887,,,,,
waveform_only,future_long_output,sklearn_hist_gradient_boosting,22,0.5144,0.6883,0.6841,0.7323,0.7529,,,,,
waveform_only,future_overheat,sklearn_hist_gradient_boosting,22,0.5905,0.5933,0.5143,0.7342,0.5643,,,,,
waveform_only,future_peak_any,sklearn_hist_gradient_boosting,22,0.2843,0.7951,0.7546,0.647,0.8613,,,,,
waveform_only,pet_state_multiclass,sklearn_hist_gradient_boosting,22,,0.6935,0.3419,,,0.2945,0.5776,,,
waveform_only,remaining_turns_research,sklearn_hist_gradient_boosting,22,,,,,,,,0.4777,0.463,0.9982
enhanced,future_end_within_5,sklearn_hist_gradient_boosting,71,0.0004,0.9997,0.8888,0.6667,0.9992,,,,,
enhanced,future_long_output,sklearn_hist_gradient_boosting,71,0.5144,0.713,0.7107,0.7388,0.7909,,,,,
enhanced,future_overheat,sklearn_hist_gradient_boosting,71,0.5905,0.6438,0.6052,0.7308,0.6814,,,,,
enhanced,future_peak_any,sklearn_hist_gradient_boosting,71,0.2843,0.828,0.7819,0.6906,0.8982,,,,,
enhanced,pet_state_multiclass,sklearn_hist_gradient_boosting,71,,0.9996,0.9992,,,0.999,0.9996,,,
enhanced,remaining_turns_research,sklearn_hist_gradient_boosting,71,,,,,,,,0.4749,0.4635,0.9982

## 解释
- `stats_only`：只用相位、近期斜率、累计量和事件密度。
- `waveform_only`：只用 NMF 权重、前缀重构误差和 NMF 续写出的未来波形摘要。
- `enhanced`：合并统计特征和波形特征，用于检验波形基是否能增强桌宠目标预测。
- `remaining_turns_research` 保留为科研指标，不建议直接作为桌宠展示目标。