# OpenCV-Python Tutorial

## Gui Features in OpenCV

### Getting Started with Images

#### 图片读取

使用**cv2.imread()**读取图片。第一个参数需要传入图片的完整路径，第二个参数指定图片的读取方式。尽管图像路径错误，它也不会抛出错误，但是输出图像时会为None。

- cv2.IMREAD_COLOR：加载彩色图像。图像的任何透明度都会被忽略。它是默认方式。可使用1表示。
- cv2.IMREAD_GRAYSCALE:在灰度模式下加载图像。可使用0表示。
- cv2.IMREAD_UNCHANGED:加载图像包括alpha通道。可使用-1表示。按此种方式读出来即存在BGRA四个通道。alpha通道是一个8位的灰度通道，$$2^8=256$$，该通道用256级灰度来记录图片中的透明信息，定义透明、不透明和半透明区域，其中白(255)表示不透明，黑(0)表示透明，灰表示半透明。

```
import numpy as np 
import cv2
img = cv2.imread('test.jpg',0)
```

在图像处理中，Alpha用来衡量一个像素或图像的透明度。在非压缩的32位RGB图像中，每个像素是由四个部分组成：一个Alpha通道和三个颜色分量(R、G和B)。当Alpha值为0时，该像素是完全透明的，而当Alpha值为255时，则该像素是完全不透明。 Alpha混色是将源像素和背景像素的颜色进行混合，最终显示的颜色取决于其RGB颜色分量和Alpha值。它们之间的关系可用下列公式来表示： 
显示颜色 = 源像素颜色 X alpha / 255 + 背景颜色 X (255 - alpha) / 255

#### 图像显示

使用函数**cv2.imshow()**在一个窗口中显示图像。窗口会自动适配图像大小。第一个参数是一个字符串表示窗口名。第二个窗口是显示的图像。你可以创建任意多窗口，但是得具有不同的窗口名。

```
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

图像显示如下：

![](.\code\gray_test.png)

**cv2.waitKey()**是一个键盘绑定函数。它的参数是时间(毫秒)。该函数为任何键盘事件等待指定的毫秒。如果你在这段时间内按了任何一个键，程序就会继续。如果传递0，它将无限期地等待击键。它也可以设置为检测特定的按键，如果按下了$$ESC$$键等，如下。

```
if cv2.waitKey(100) == 27:
    print 'wait 100 ms'
    pass
'''
等待用户触发事件,等待时间为100ms，
如果在这个时间段内, 用户按下ESC(ASCII码为27),执行if体
如果没有按，if函数不做处理
'''
```

**cv2.destroyAllWindows()**简单地销毁我们创建的所有窗口。如果您想销毁任何特定的窗口，请使用函数**cv2.destroyWindow()**，在该函数中，您将传递确切的窗口名作为参数。

**注意**：有一种特殊情况，您可以创建一个窗口并在以后加载图像到其窗口中。在这种情况下，您可以指定窗口是否可调整大小。这是通过函数**cv2.namedWindow()**完成的。默认情况下，标志是**cv2.WINDOW_AUTOSIZE**。但是如果你指定flag为**cv2.WINDOW_NORMAL**，可以调整窗口大小。这将有助于当图像太大的尺寸和添加跟踪栏到窗口。

#### 图像保存

使用函数**cv2.imwrite()**保存图像。

第一个参数是保存图像文件名，第二个参数是想要保存的图像。

```
cv2.imwrite("gray_test.png",img)
```

#### 总结

下面程序按灰度模式加载图像，显示它。如果按键’s'，保存图片并退出。如果按‘ESC’不保存图像退出。

```
import numpy as np
import cv2
img = cv2.imread('test.jpg',0)
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27: # wait for ESC key to exit
cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
cv2.imwrite('test_gray.png',img)
cv2.destroyAllWindows()
```

**Warning:**如果你使用64位机器，需要将**k=cv2.waitKey(0)**改为**k=cv2.waitKey(0) & 0xFF**

#### 使用Matplotlib

Matplotlib是一个用于Python的绘图库，它提供了各种各样的绘图方法。您将在以后的文章中看到它们。在这里，您将学习如何使用Matplotlib显示图像。你可以使用Matplotlib缩放图像，保存它等。

- plt.imshow()：显示图像
- plt.imsave():保存图像
- plt.imread():读取图像

如下显示不正确

```
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('test.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()
```

Matplotlib中有很多绘图选项。详情请参考Matplotlib文档。

**警告：**OpenCV加载的彩色图像处于BGR模式。但是Matplotlib以RGB模式显示。因此，如果使用OpenCV读取图像，在Matplotlib中将不能正确显示彩色图像。详情请参阅练习。

正确显示方法：

```
import numpy as np 
import cv2
from matplotlib import pyplot as plt 
img = cv2.imread('test.jpg',1)
b,g,r = cv2.split(img)
new_img = cv2.merge([r,g,b])
plt.imshow(new_img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()
```

### Getting Started with Videos

### Drawing Functions in OpenCV(绘制函数)

#### 目标

- 学习使用OpenCV绘制不同的几何图形
- 将会学习这些函数：**cv2.line()、cv2.circle()、cv2.rectangle()、cv2.ellipse()、cv2.putText()**等。

#### 编码

在上述所有函数中，您将看到如下所示的一些常见参数：

- img:你想要绘制图形的图像
- color:形状的颜色。对于BGR，将它作为一个元组传递，例如:(255,0,0)for blue。对于灰度，只需传递标量值。
- thickness:线或圆的厚度等。如果将-1传递给封闭的图形(如圆圈)，它将填充形状。默认厚度= 1。
- lineType:线条类型。

#### 画线

要画一条线，你需要传递线的起始和结束坐标。我们将创建一个黑色的图像，并在它上面从左上角到右下角画一条蓝线。

```
import numpy as np
import cv2
# Create a black image
img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 画矩形

要画一个长方形，你需要长方形的左上角和右下角。这次我们将在图像的右上角画一个绿色矩形。

```
img = cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
```

#### 画圆

要画一个圆，你需要它的中心坐标和半径。我们将在上面画的矩形内画一个圆。

```
img = cv2.circle(img,(447,63), 63, (0,0,255), -1)
```

#### 画椭圆（Drawing Ellipse）

为了绘制椭圆，我们需要传递几个参数。一个参数是中心位置(x,y)下一个参数是轴长度(长轴，短轴)。角是椭圆逆时针方向旋转的角度。起始角和结束角表示从主轴顺时针方向测量的椭圆圆弧的起始和结束。也就是说，赋值0和360得到完整的椭圆。要了解更多细节，请查看**cv2.ellipse()**的文档。下面的例子在图像的中心画了一个半椭圆。

```
img = cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
```

#### 绘制多边形（Drawing Polygon）

要画一个多边形，首先需要顶点的坐标。将这些点放入形状ROWSx1x2的数组中行是顶点的数量，它的类型应该是int32。这里我们用黄色画一个有四个顶点的小多边形。

```
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
img = cv2.polylines(img,[pts],True,(0,255,255))
```

**注意：**如果第三个参数是False，你将得到一个折线连接所有的点，而不是一个闭合的形状。

**注意：**polylines()可用于绘制多条线。只需创建一个要绘制的所有线条的列表，并将其传递给函数。所有的线将单独绘制。为每一行调用**cv2.line()**比绘制一组行更好更快。

#### 图像中添加文本



## Core Operations

### Basic Operations on Images

##  目标检测

### Face Detection using Haar Cascades

