---
    title：模型优化总结
---

### 1、统计版本
- 原理
    - 直接基于股票10分钟前的波动指数给出后10分钟内的波动指数
- 得分
  - 0.32


### 2、LGB版本
- 原理
  - 构建股票在10分钟内的特征，使用optuna寻找最优参数，并使用交叉验证的方式进行预测
- 特征
  - 订单（滑窗150、300、450）
    - 广度特征
      - bid
      - ask
      - price
    - 加权容量、价格
      - wap
      - total_volume
      - volume_imblance
    - 波动指数
      - log_return
      - log_return2
      - log_return3
  - 交易
    - 总和
    - 次数
    - 波动指数
