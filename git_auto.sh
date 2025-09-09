#!/bin/bash

# --- 脚本开始 ---

# 1. 设置提交信息
# 检查用户是否提供了自定义信息（第一个参数 $1）
if [ -z "$1" ]
  then
    # 如果没有提供，就使用当前日期作为默认信息
    commit_message=$(date "+%Y-%m-%d Auto-commit")
  else
    # 如果提供了，就使用用户的信息
    commit_message="$1"
fi

echo "================================================"
echo "Starting Git Auto-Sync..."
echo "Commit message will be: \"$commit_message\""
echo "================================================"

# 2. 执行核心 Git 命令
# 使用 && 确保按顺序、按条件执行

# 添加所有更改
echo "Step 1: Adding all files..."
git add -A && \

# 提交更改
echo "Step 2: Committing changes..."
git commit -m "$commit_message" && \

# 先从远程拉取更新，进行同步
echo "Step 3: Pulling from remote to sync..."
git pull origin master && \

# 推送到远程仓库
echo "Step 4: Pushing to remote..."
git push origin master

# 3. 检查上一步的退出码，判断是否成功
if [ $? -eq 0 ]; then
    echo "================================================"
    echo "Sync successful!"
    echo "================================================"
else
    echo "================================================"
    echo "Sync failed at some step. Please check the output above."
    echo "================================================"
fi

# 4. 暂停等待用户按键
echo "Press any key to exit..."
read -n 1 -s