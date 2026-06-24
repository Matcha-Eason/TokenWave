# GitHub Setup

本项目建议作为独立科研仓库上传，不和 Codex 桌宠插件仓库混在一起。

## 1. 在 GitHub 网页创建仓库

建议仓库名：

```text
dialogue-resonance-curve
```

创建时建议：

- Visibility：按需要选择 Public 或 Private。
- 不要勾选自动创建 README。
- 不要勾选自动创建 `.gitignore`。
- 不要勾选自动创建 LICENSE，除非已经确定开源协议。

## 2. 本地初始化 Git

在本目录执行：

```bash
git init
git branch -M main
```

检查将要提交的文件：

```bash
git status --short
git check-ignore -v outputs/coderforge_full/turn_timeseries.parquet
git check-ignore -v plugins/codex-pet-state-tracker/README.md
```

上面两个 `git check-ignore` 都应该显示被 `.gitignore` 忽略。

## 3. 添加文件

```bash
git add README.md requirements.txt .gitignore assets docs results scripts
git status --short
```

确认没有以下内容：

```text
data/
outputs/
.venv/
plugins/
dist/
.agents/
.github/
*.parquet
```

## 4. 提交

```bash
git commit -m "Initial research repository"
```

## 5. 绑定远程仓库

把 URL 换成你自己的 GitHub 仓库地址：

```bash
git remote add origin git@github.com:<your-name>/dialogue-resonance-curve.git
```

如果你用 HTTPS：

```bash
git remote add origin https://github.com/<your-name>/dialogue-resonance-curve.git
```

## 6. 上传

```bash
git push -u origin main
```

## 7. 可选：上传大结果文件

本仓库默认不上传全量 parquet。如果你想共享大文件，建议不要放进 Git 历史，而是使用：

- GitHub Release 附件
- Hugging Face Dataset
- ModelScope Dataset
- Google Drive / OneDrive
- Git LFS

建议至少不要直接提交：

```text
outputs/coderforge_full/turn_timeseries.parquet
outputs/coderforge_full/trajectory_wave_features.parquet
outputs/*/*examples.parquet
outputs/*/*rows.parquet
```

## 8. 推荐上传后的检查

```bash
git ls-files | grep -E '(^outputs/|^data/|^plugins/|^dist/|\\.parquet$)' || true
```

如果没有输出，说明大数据和插件文件没有进仓库。
