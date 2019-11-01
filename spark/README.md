# Spark
## Spark SQL

Spark SQL是一个用于结构化数据处理的Spark模块。与基本的Spark RDD API不同，Spark SQL提供的接口为Spark提供了有关数据结构和正在执行的计算的更多信息。在内部，Spark SQL使用这些额外的信息来执行额外的优化。有几种与Spark SQL交互的方法，包括SQL和Dataset API。当计算结果时，使用相同的执行引擎，而不依赖于使用哪种API/语言来表示计算。这种统一意味着开发人员可以很容易地在不同的api之间来回切换，而这些api提供了表达给定转换的最自然的方式。

