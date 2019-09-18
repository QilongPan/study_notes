# introduction-to-algorithm
# 算法在计算中的作用

非形式地说，**算法**就是任何良定义的计算过程，该过程取某个值或值得集合作为输入并产生某个值或值的集合作为输出。

# 算法基础

## 插入排序

```
def insertion_sort(arr,descent):
    for j in range(1,len(arr)):
        key = arr[j]
        i = j -1
        if descent:
            while i >= 0 and arr[i] < key:
                arr[i+1] = arr[i]
                i = i - 1
        else:
            while i >= 0 and arr[i] > key:
                arr[i+1] = arr[i]
                i = i - 1
        arr[i+1] = key
```

插入排序最坏情况运行时间为$$\Theta(n^{2})$$。插入排序是稳定的。

## 归并排序

```
def merge_sort(arr,left,right,descent):
    if left < right:
        middle = (right + left) // 2
        merge_sort(arr,left,middle,descent)
        merge_sort(arr,middle+1,right,descent)
        merge(arr,left,middle,right,descent)

def merge(arr,left,middle,right,descent):
    left_arr_len = middle - left + 1
    right_arr_len = right - middle
    left_arr = []
    right_arr = []
    for i in range(left_arr_len):
        left_arr.append(arr[left+i])
    for i in range(right_arr_len):
        right_arr.append(arr[middle+1+i])
    i = 0
    j = 0
    k = left
    while i < left_arr_len and j < right_arr_len:
        if descent:
            if left_arr[i] >= right_arr[j]:
                arr[k] = left_arr[i]
                i = i + 1
            else:
                arr[k] = right_arr[j]
                j = j + 1               
        else:
            if left_arr[i] <= right_arr[j]:
                arr[k] = left_arr[i]
                i = i + 1
            else:
                arr[k] = right_arr[j]
                j = j + 1        
        k = k + 1
    while i < left_arr_len:
        arr[k] = left_arr[i]
        i += 1
        k += 1
    while j < right_arr_len:
        arr[k] = right_arr[j]
        j += 1
        k += 1    
```

时间复杂度为$$\Theta(nlgn)$$。归并排序是稳定的。

# 函数的增长

## 渐进记号

### $\theta$记号

$$\theta$$记号给出一个函数的上界和下界。

对一个给定的函数$$g(n)$$，用$$\theta(g(n))$$来表示以下函数的集合：

$$\theta(g(n))=$${$$f(n)：存在正常量c_{1}、c_{2}和n_{0},使得对所有n\geq n_{0},有0\leq c_{1}g(n)\leq f(n)\leq c_{2}g(n)$$}

### $O$记号

当只有一个渐近上界时，使用$$O$$记号。

对一个给定的函数$$g(n)$$，用$$Og(n)$$来表示以下函数的集合：

$$O(g(n))=$${$$f(n):存在正常量c和n_{0}，使得对所有n\geq n_{0},有0\leq f(n)\leq cg(n)$$}。

### $\Omega $记号

当只有一个渐近下界时，使用$$\Omega$$记号。

$$\Omega(g(n))=$${$$f(n):存在正常量c和n_{0}，使得对所有n\geq n_{0},有0\leq  cg(n)\leq f(n)$$}。

### $o$记号

由$$O$$记号提供的渐近上界可能是也可能不是渐近紧确的。界$$2n^2=O(n^2)$$是渐近紧确的，但是界$$2n=O(n^2)$$却不是。形式化地定义$$o(g(n))$$为以下集合：

$$o(g(n))=$${$$f(n):对任意正常量c> 0，存在常量n_{0}>0，使得对所有n\geq n_{0},有0\leq f(n)\leq cg(n)$$}。例如，$$2n=o(n^2)$$,但是$$2n^2\neq o(n^2)$$.

### $w$记号

$$w$$记号与$$\Omega $$记号的关系类似于$$o$$记号与$$O$$记号的关系。

$$w(g(n))=$${$$f(n):对任意正常量c> 0，存在常量n_{0}>0，使得对所有n\geq n_{0},有0\leq cg(n)\leq f(n) $$}。例如，$$n^2/2=w(n)$$,但是$$n^2/2\neq w(n^2)$$。

# 分治策略

在分治策略中，我们递归地求解一个问题，在每层递归中应用如下三个步骤：

- **分解**步骤将问题划分为一些子问题，子问题的形式与原问题一样，只是规模更小。
- **解决**步骤递归地求解出子问题。如果子问题的规模足够小，则停止递归，直接求解。
- **合并**步骤将子问题的解组合成原问题的解。

## 最大子数组问题

案例:

```
13 -3 -25 20 -3 -16 -23 18 20 -7 12 -5 -22 15 -4 7
```

最大子数组为18 20 -7 12

### 使用分治策略的求解办法

假定我们要寻找子数组$$A[low..high]$$的最大子数组。使用分治技术意味着我们要将子数组划分为两个规模尽量相等的子数组。也就是说，找到子数组的中央位置，比如mid，然后考虑求解两个子数组$$A[low..mid]$$和$$A[mid+1..high]$$。$$A[low..high]$$的任何连续子数组$$A[i..j]$$所处的位置必然是以下三种情况之一：

- 完全位于子数组$$A[low..mid]$$中
- 完全位于子数组$$A[mid+1..high]$$中
- 跨越了中点

```
def maximum_subarray(arr,low,high):
    if low == high:
        return low,high,arr[low]
    else:
        mid = (low+high)//2
        left_low,left_high,left_sum = maximum_subarray(arr,low,mid)
        right_low,right_high,right_sum = maximum_subarray(arr,mid+1,high)
        cross_low,cross_high,cross_sum = find_max_crossing_subarray(arr,low,mid,high)
        if left_sum >= right_sum and left_sum >= cross_sum:
            return left_low,left_high,left_sum
        elif right_sum >= left_sum and right_sum >= cross_sum:
            return right_low,right_high,right_sum
        else:
            return cross_low,cross_high,cross_sum

def find_max_crossing_subarray(arr,low,mid,high):
    arr_sum = 0
    left_sum = None
    max_left = None
    for i in range(mid,low-1,-1):
        arr_sum += arr[i]
        if left_sum == None:
            left_sum = arr_sum
            max_left = i
        else:
            if arr_sum > left_sum:
                left_sum = arr_sum
                max_left = i 
    right_sum = None
    arr_sum = 0
    max_right = None
    for i in range(mid+1,high+1):
        arr_sum += arr[i]
        if right_sum == None:
            right_sum = arr_sum
            max_right = i
        else:
            if arr_sum > right_sum:
                right_sum = arr_sum
                max_right = i
    return max_left,max_right,left_sum+right_sum
```

时间复杂度为$$\Theta(nlgn)$$。

## 矩阵乘法的$Strassen$算法

为简单起见，当使用分治算法计算矩阵积$$C=A\cdot B​$$时，假定三个矩阵均为$$n\times n​$$矩阵，其中n为2的幂。当n不为2的幂时，需要进行变形。
$$
A = \begin{bmatrix}
A_{11} & A_{12}\\ 
A_{21} & A_{22}
\end{bmatrix},B=\begin{bmatrix}
B_{11} &B_{12} \\ 
B_{21} & B_{22}
\end{bmatrix},C = \begin{bmatrix}
C_{11} &C_{12} \\ 
C_{21} & C_{22}
\end{bmatrix} \\
$$
因此可以将公式$$C=A\cdot B$$改写为：
$$
\begin{bmatrix}
C_{11} &C_{12} \\ 
C_{21} & C_{22}
\end{bmatrix}=\begin{bmatrix}
A_{11} & A_{12}\\ 
A_{21} & A_{22}
\end{bmatrix}\cdot \begin{bmatrix}
B_{11} &B_{12} \\ 
B_{21} & B_{22}
\end{bmatrix}\\
$$
等价于如下4个公式：
$$
\begin{aligned}
C_{11}=A_{11}\cdot B_{11}+A_{12}\cdot B_{21}\\
C_{12}=A_{11}\cdot B_{12}+A_{12}\cdot B_{22}\\
C_{21}=A_{21}\cdot B_{11}+A_{22}\cdot B_{21}\\
C_{22}=A_{21}\cdot B_{12}+A_{22}\cdot B_{22}
\end{aligned}
$$
$$strassen$$算法之所以要n为2的幂的原因在于矩阵二分后，两个子矩阵加减法要满足相同规模。

当n不为2的幂时:把矩阵补全成为2的幂次规模即可。由于矩阵乘法性质，就算扩大矩阵（补0），也会保留原有的结果，而其他部分为0，也就是说算完之后再从结果矩阵将需要部分扣下来即可。


# 概率分析和随机算法

当分析一个随机算法的运行时间时，我们以运行时间的期望值衡量，其中输入值由随机数生成器产生。我们将一个随机算法的运行时间称为期望运行时间。

## 雇用问题

**概率分析：**我们对所有可能输入产生的运行时间取平均。当报告此种类型的运行时间时，我们称其为平均情况运行时间。

## 指示器随机变量

# 堆排序

堆是一个数组，它可以被看成一个近似的完全二叉树。树上的每一个结点对应数组中的一个元素。除了最底层外，该树是完全充满的，而且是从左向右填充。

在最大堆中，最大堆性质是指除了根以外的所有结点$$i$$都要满足：
$$
A[PARENT(i)]\geq A[i]
$$
最小堆性质是除了根以外的所有结点$$i$$都有
$$
A[PARENT(i)]\leq  A[i]
$$

## 维护堆的性质



# 中位数和顺序统计量

第$$i$$个顺序统计量是该集合中第$$i$$小的元素。最小值是第1个顺序统计量，最大值是第$$n$$个顺序统计量。

本章将讨论从一个由$$n$$个互异的元素构成的集合中选择第$$i$$个顺序统计量的问题。

## 最小值和最大值

同时找到最小值和最大值，下面有两种方式，第二种方式更优：

- 分别独立的找出最小值和最大值，这各需要$$n-1$$次比较，共需$$2n-2$$次比较。
- 记录已知的最小值和最大值，但我们并不是将每一个输入元素与当前的最小值和最大值进行比较，而是对输入元素成对的进行处理。首先，我们将一对输入进行相互比较，然后将较小的与当前最小值进行比较，把较大的与当前最大值进行比较。这样，对每两个元素需$$3$$次比较，所以总共需要$$3\left \lfloor \frac{n}{2} \right \rfloor$$ 次比较。

## 期望为线性时间的选择算法

