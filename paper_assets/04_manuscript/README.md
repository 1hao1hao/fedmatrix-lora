# 当前论文

| 文件 | 用途 |
|---|---|
| `main.tex` / `main.pdf` | 当前英文主稿及编译 PDF；英文稿是内容源文件。 |
| `main_zh.tex` / `main_zh.pdf` | 按当前英文 P2 主稿同步的中文阅读版；公式、图表、标签和参考文献顺序与英文稿一致。 |
| `figures/` | 两版共同引用的 Fig. 1--6 矢量图；图内英文标注保持原图。 |
| `IEEEtran.cls` | 编译所需的 IEEE 模板。 |

在仓库根目录执行 `bash scripts/compile_infocom_paper.sh all` 可重新编译两版；单独编译用 `en` 或 `zh`。英文用 pdfLaTeX，中文用 XeLaTeX。2026-10-28 P2 修订后，英文 PDF 为 13 页，中文 PDF 为 11 页。Fig. 1 已替换为 P2 配对收益图。由于中文字符排版密度不同，两版页数不相同，但章节、公式、图表与引用结构一致。

`infocom_2027/` 中保留导师讨论材料、参考文献核验表和历史备份；它们不是当前论文的编辑入口。此前特意保存的投稿前快照没有删除。

当前英文稿是仓库内已确认的最新活动版本；由于未取得投稿系统中实际上传的 PDF，不能断言它与当时的投稿文件逐字相同。

Status: the English P2 rewrite and its Chinese reading version are synchronized.
