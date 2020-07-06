# git笔记

[git教程]( https://git-scm.com/book/zh/v2 )

git config --list  查看config配置

 git config --global --edit  编辑config配置

git help <verb> 使用案例git help config，用于获取帮助

git clone 克隆远程仓库 

git clone https://github.com/libgit2/libgit2 mylibgit 其中mylibgit为仓库的重命名

git status 查看哪些文件处于什么状态

git分为工作区、暂存区、已暂存区

git add file.txt 可以让git跟踪文件

git restore --staged <file>...取消放入暂存区

git rm --cached file 将文件移除暂存区，即git不跟踪，但不从本地目录中删除

git rm -f file 将文件移除暂存区，并从本地目录删除

git log查看日志，按q即可退出

