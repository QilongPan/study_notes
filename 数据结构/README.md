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

$$w(g(n))=$${$$f(n):对任意正常量c> 0，存在常量n_{0}>0，使得对所有n\geq n_{0},有0\leq cg(n)\leq f(n) $$}。例如，$$n^2/2=w(n)$$,但是$$n^2/2\neq w(n^2)$$.

# 概率分析和随机算法

## 雇用问题

**概率分析：**我们对所有可能输入产生的运行时间取平均。当报告此种类型的运行时间时，我们称其为平均情况运行时间。

## 指示器随机变量



# 中位数和顺序统计量

第$$i$$个顺序统计量是该集合中第$$i$$小的元素。最小值是第1个顺序统计量，最大值是第$$n$$个顺序统计量。

本章将讨论从一个由$$n$$个互异的元素构成的集合中选择第$$i$$个顺序统计量的问题。

## 最小值和最大值

同时找到最小值和最大值，下面有两种方式，第二种方式更优：

- 分别独立的找出最小值和最大值，这各需要$$n-1$$次比较，共需$$2n-2$$次比较。
- 记录已知的最小值和最大值，但我们并不是将每一个输入元素与当前的最小值和最大值进行比较，而是对输入元素成对的进行处理。首先，我们将一对输入进行相互比较，然后将较小的与当前最小值进行比较，把较大的与当前最大值进行比较。这样，对每两个元素需$$3$$次比较，所以总共需要$$3\left \lfloor \frac{n}{2} \right \rfloor$$ 次比较。

## 期望为线性时间的选择算法

