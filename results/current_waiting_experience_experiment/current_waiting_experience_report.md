# 当前等待体验实验

本实验把目标从“未来几轮”拉近到“当前正在输出的这一轮”。由于 CoderForge 没有真实流式时间戳，脚本把当前轮的模型输出和思考长度切成 0%、25%、50%、75%、100% 五个前缀，模拟桌宠在输出过程中的可见信息。

## 实验设置

- 样本数：180,000
- 轨迹数：3,000
- 对照组：只看历史统计、历史统计 + 当前流式前缀、规则状态机。
- 目标：当前等待体验状态、当前是否深度输出、当前是否红温调试。

## 阈值

- 长模型输出阈值：297.0
- 长思考规划阈值：120.0
- 高环境反馈阈值：1204.0
- 深度可见负载阈值：383.0
- 红温压力阈值：7.94

## 整体状态识别

model,target,prefix_fraction,features,accuracy,balanced_accuracy,macro_f1,weighted_f1,positive_rate,f1,auc
rule_baseline,current_wait_state,all,0,0.6894,0.5483,0.5468,0.6958,,,
history_only,current_wait_state,all,79,0.8277,0.8216,0.7846,0.8346,,,
history_plus_stream,current_wait_state,all,88,0.9374,0.9272,0.9038,0.9405,,,

## 流式前缀增益

model,target,prefix_fraction,features,accuracy,balanced_accuracy,macro_f1,weighted_f1,positive_rate,f1,auc
history_only,current_wait_state,0.0,79,0.8277,0.8216,0.7846,0.8346,,,
history_only,current_wait_state,0.25,79,0.8277,0.8216,0.7846,0.8346,,,
history_only,current_wait_state,0.5,79,0.8277,0.8216,0.7846,0.8346,,,
history_only,current_wait_state,0.75,79,0.8277,0.8216,0.7846,0.8346,,,
history_only,current_wait_state,1.0,79,0.8277,0.8216,0.7846,0.8346,,,
history_plus_stream,current_wait_state,0.0,88,0.875,0.8586,0.8181,0.8776,,,
history_plus_stream,current_wait_state,0.25,88,0.9492,0.9431,0.9212,0.9519,,,
history_plus_stream,current_wait_state,0.5,88,0.9573,0.9495,0.9296,0.959,,,
history_plus_stream,current_wait_state,0.75,88,0.9529,0.9432,0.9242,0.9543,,,
history_plus_stream,current_wait_state,1.0,88,0.9523,0.9417,0.9239,0.9534,,,

## 当前二分类目标

model,target,prefix_fraction,features,accuracy,balanced_accuracy,macro_f1,weighted_f1,positive_rate,f1,auc
history_only,is_deep_output,all,79,0.9069,0.8036,,,0.1803,0.7132,0.954
history_only,is_overheat_debugging,all,79,0.9536,0.9452,,,0.4042,0.9401,0.984
history_plus_stream,is_deep_output,all,88,0.9824,0.9629,,,0.1803,0.9503,0.9982
history_plus_stream,is_overheat_debugging,all,88,0.9994,0.9995,,,0.4042,0.9993,1.0

## 标签分布

state,count,state_zh,rate
overheat_debugging,72395,红温调试,0.4022
steady_work,38655,稳定工作,0.2148
deep_output,33175,深度输出,0.1843
reading_understanding,23430,读题理解,0.1302
closing,12345,收束,0.0686

## 结论

- 只看历史统计主要反映任务惯性，适合识别红温调试这类有上下文延续性的状态。
- 加入当前流式 token 后，深度输出识别会更贴近桌宠的真实等待体验，因为用户真正感知到的是当前输出正在变长。
- 规则状态机可以作为第一版插件 baseline；机器学习模型适合做离线校准和阈值学习。
- 这版目标比“未来 3/5 轮”更适合桌宠，因为输出可以在本轮内随 token 和事件持续更新。