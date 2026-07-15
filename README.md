# alphaA

量化选股、滚动机器学习与A股回测工程。

- [`code/BackTest/README.md`](code/BackTest/README.md)：回测框架、网页运行方式与交易规则。
- [`code/Model/README.md`](code/Model/README.md)：滚动模型、PCA风格轮动、行业中性优化器与实时训练页面。

## 启动

```powershell
cd code\Model
pip install -r requirements.txt
python web_app.py
```

浏览器访问 `http://127.0.0.1:8090`。

## 数据

程序从本地 `BackTestData/` 读取因子、价格、复权、票池、ST和基准数据。完整
`factordata.parquet`、`twap.parquet` 以及运行产生的 `output/` 不提交到GitHub；
克隆后请将本地数据放回该目录。行业中性优化器使用的行业分类文件可在模型网页上传。
