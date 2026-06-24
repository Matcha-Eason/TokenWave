# 桌宠状态预测实验

- 样本数：120,000
- 轨迹数：9,231
- 训练/测试样本：89,996 / 30,004
- 特征数：52

## 预测指标
task,mae_turns,median_abs_error_turns,r2_log_remaining,positive_rate,accuracy,balanced_accuracy,f1,auc,macro_f1,weighted_f1
remaining_turns_regression,10.8326,7.3956,0.7548,,,,,,,
future_end_within_5,,,,0.1011,0.9523,0.8397,0.7475,0.9787,,
future_long_output_3,,,,0.4185,0.7787,0.783,0.7538,0.8606,,
future_overheat_5,,,,0.5227,0.6536,0.6507,0.6831,0.7165,,
pet_state_multiclass,,,,,0.9475,0.9399,,,0.9225,0.9506

## 状态标签分布
state,count,state_zh,rate
overheat_debugging,51771,红温调试,0.4314
deep_output,28437,深度输出,0.237
reading_understanding,20207,读题理解,0.1684
steady_work,10295,稳定工作,0.0858
closing,9290,收束,0.0774

## 状态定义
- 读题理解：早期阶段，尚未进入强输出或红温。
- 稳定工作：中间常规推进。
- 深度输出：当前或最近 5 轮存在长输出压力。
- 红温调试：近期 5 轮错误/测试压力高。
- 收束：预计 3 轮内结束。