# 当前论文

| 文件 | 用途 |
|---|---|
| `main.tex` / `main.pdf` | 当前英文主稿及编译 PDF；英文稿是内容源文件。 |
| `main_zh.tex` / `main_zh.pdf` | 补丁前的中文阅读版；本次 P2 英文修订未同步到中文稿。 |
| `figures/` | 英文稿当前使用 Fig. 1--6；中文稿仍使用补丁前的图。 |
| `IEEEtran.cls` | 编译所需的 IEEE 模板。 |

在仓库根目录执行 `bash scripts/compile_infocom_paper.sh en` 可重新编译英文稿。英文用 pdfLaTeX，中文用 XeLaTeX。2026-10-28 P2 修订后，英文 PDF 为 13 页；中文 PDF 仍为此前的 10 页。英文 Fig. 1 已替换为 P2 配对收益图。

`infocom_2027/` 中保留导师讨论材料、参考文献核验表和历史备份；它们不是当前论文的编辑入口。此前特意保存的投稿前快照没有删除。

当前英文稿是仓库内已确认的最新活动版本；由于未取得投稿系统中实际上传的 PDF，不能断言它与当时的投稿文件逐字相同。

Status: the English P2 rewrite is applied; the Chinese reading version has not been synchronized by this patch.
