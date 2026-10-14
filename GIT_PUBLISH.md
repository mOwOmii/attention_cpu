# Git 与 GitHub 提交

当前受限会话执行 `git init -b main` 返回 `.git/branches/: Read-only file system`，尚未创建 Git 提交或 GitHub 仓库。发布脚本需在普通终端执行。

## 初始化、本地提交和 GitHub 私有仓库

```bash
cd /share/home/renhj/attention_cpu
bash scripts/publish_github.sh
```

脚本使用仓库级 Git 配置：姓名 `renhj`，邮箱 `2677703123@qq.com`。姓名可通过 `ATTENTION_GIT_NAME` 设置。提交源码、测试、配置、脚本及项目文档；排除 `runs/`、Python 缓存、本地 agent 元数据和凭据文件。

GitHub 仓库默认名为 `attention_cpu`，使用当前登录账号并创建私有仓库。可通过 `ATTENTION_GITHUB_REPO` 改名。脚本使用 GitHub CLI `gh` 的网页授权，不读取或保存账号密码。若尚未安装 `gh`，脚本会完成本地提交后停止；安装后重新执行即可继续发布。

仅本地提交：

```bash
bash scripts/publish_github.sh --local-only
```

已有仓库不会被覆盖，不强制推送；若 origin 指向其他仓库或同名远端为公开仓库，脚本停止并说明原因。运行数据中保存的验收报告和 trace 不在提交范围内，README 中对应 `runs/` 链接需在产生了这些结果的工作目录查看。
