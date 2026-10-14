"""Readable data card and process report, generated from actual execution receipts."""
import argparse,html,json,collections
from pathlib import Path
from build_catalog import loadjl,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root)
    def get(n):return json.loads((root/n).read_text())
    cat=get('audit/CATALOG_RECEIPT.json');clean=get('audit/CLEAN_RECEIPT.json');split=get('audit/SPLIT_RECEIPT.json');accept=get('audit/ACCEPTANCE.json')
    lines=['# PepSeek：两类监督数据集','', '本版将原联合样本拆入两张监督表，同时保存配对身份、原始记录和已核验关联。数据表的拆分不代表增加独立配对。','',
           '## 数据规模','', '| 数据 | 来源整理后的候选 | 最终 clean |','|---|---:|---:|',f"| 序列＋定量 | {cat['quantitative_observations']:,} | {clean['quantitative_observations']:,} |",f"| 序列＋结构 | {cat['structure_observations']:,} | {clean['structural_observations']:,} |",'',
           '## 实际分区','', '| 分区 | 模型输入配对 | 定量观察 | 结构观察 | 去重结构视图 | 连通分组 |','|---|---:|---:|---:|---:|---:|']
    for s,r in clean['split_counts'].items():lines.append(f"| {s} | {r['input_pairs']:,} | {r['quantitative_observations']:,} | {r['structural_observations']:,} | {r['distinct_structure_views']:,} | {r['components']:,} |")
    lines+=['','实际比例由完整连通组决定。本版最大组包含大多数观察，整体放入 train；验证和测试保留其余独立组。以下逐终点数量决定各项评估能够支持的范围。','', '| 定量终点 | train | valid | test |','|---|---:|---:|---:|']
    for ep in ['pKd','pKi','pIC50_assay_dependent','pEC50','pKm']:
        lines.append('| '+ep+' | '+' | '.join(str(clean['split_counts'][s]['endpoints'].get(ep,0)) for s in ['train','valid','test'])+' |')
    lines+=['','本版 pEC50 测试只有1条，不能计算相关系数；pKm 没有留出测量。Kd/IC50、结构接触和界面评估有较完整覆盖，Ki 的留出规模仍小。', '', '| 结构来源 | 候选观察 | clean 观察 |','|---|---:|---:|']
    before=collections.Counter(r['source'] for r in loadjl(root/'supervision/structural.jsonl'));after=collections.Counter(r['source'] for r in loadjl(root/'supervision/structural_clean.jsonl'))
    for source in sorted(before):lines.append(f'| {source} | {before[source]:,} | {after[source]:,} |')
    lines+=['','## 独立资源库','',f"- 蛋白序列 clean 库：{clean['unique_protein_sequences_clean']:,} 条去重序列（候选库 {cat['unique_protein_sequences']:,} 条）；多链蛋白另外保存组成关系。",f"- 肽序列 clean 库：{clean['unique_peptide_sequences_clean']:,} 条去重序列（候选库 {cat['unique_peptide_sequences']:,} 条）；化学身份与序列身份分开。",f"- 蛋白结构库：{clean['unique_protein_structures']:,} 个去重、映射后的结构链视图。",f"- 肽结构库：{clean['unique_peptide_structures']:,} 个去重、映射后的结构链视图。",'',
      '## 清洗和划分过程','',
      '1. 从封存清理结果读取源记录，保留原始工作簿、排除记录、单位、终点、实验来源和修正证据。',
      '2. 以序列、化学身份和蛋白组成建立稳定 ID；原联合类的 51 条带证据等级的测量—结构关联保留在 verified_links.jsonl，数据库一致性与原文构建体确认分别标明。',
      '3. 对化学图完成手性一致的原子—残基映射。未确认化学身份的结构肽显式关闭化学图与构象输入。',
      '4. 从核验过 SHA 的坐标重新生成接触、界面及距离标签；残基缺失和不完整原子以 mask 处理。',
      '5. 实际运行 MMseqs2 蛋白同源搜索和 BLASTP-short 肽相似性搜索。蛋白至少30%一致性并覆盖较短链80%；肽至少50%一致性并覆盖双方各80%。',
      '6. 同源簇、相同化学实体、同配对、相同PDB、来源文献、已核验联合关联连接后整体划分，目标80/10/10。大分组不拆开，实际数量以上表为准。',
      '7. 每个配对每轮采样一次，定量观察在配对内平均损失，多个结构视图轮换；验证和测试保留所有结构视图并按配对汇总。',
      '8. 完成所有模型标签、化学原子映射、文件哈希及分区交叉检查后封存。', '',
      '## 数据结构','', '```text','raw/                  原始表与封存源记录','supervision/          quantitative / structural 两张监督表及关联表','libraries/            蛋白/肽序列、化学实体、结构链及组成关系','coordinates/          按SHA存放的原始复合物坐标','maps/                 链、残基及组装操作映射','labels/               接触、界面和距离标签，缺失有mask','splits/               比对连接、同源簇、连通分组和分区','clean/                可直接被PepSeek读取的模型数据集','audit/                每条记录去向、检查回执和覆盖统计','```','',
      '`clean/pairs.jsonl` 每行有 `id / protein_id / peptide_id / split / component_id / affinity[] / structure_views[]`；两类监督通过同一行关联。`affinity` 保存端点与对数浓度上下界，`structure_views` 指向 NPZ 标签及映射。', '',
      '序列库为 FASTA＋JSONL。结构库每条为一个已映射链视图的 NPZ：重原子坐标（Å）、原子名、原始单体名、原子到残基索引、observed_mask、complete_mask。索引 JSONL 连接结构、序列、原始坐标 SHA 和残基映射；同一序列可有多个结构。完全未解析的链仅保留在复合物 mask 中，不计为一个可用结构。','',
      '## 训练使用','', '实验复合物坐标用于监督，不进入模型的序列预测路径。输入为各蛋白链的序列、肽序列，以及已确认的肽化学图；可选构象由该化学图独立生成。', '',
      '定量终点分为 pKd、pKi、实验依赖的 pIC50、pEC50 和 pKm。Km 单独作为酶动力学终点，不与结合亲和力混合。结构共现可作为检索正关联，但未知配对不会被标成实验阴性。','',
      '历史分区文件保持原样。本版是新数据版本，既有项目模型已接触部分来源数据；使用本版重新训练的新模型可作内部留出评估，新的外部确认需要另外锁定独立样本。','',
      '## 验收','',f"- 模型数据 SHA256：`{accept['model_dataset_sha256']}`",f"- 跨分区连通分组：{accept['cross_partition_components']}",'- 本次完成数据准备与软件验收，未启动正式科学训练。','',
      '## clean 集额外隔离原因','', '```json',json.dumps(clean['clean_holds'],ensure_ascii=False,indent=2),'```']
    text='\n'.join(lines)+'\n';(root/'README_ZH.md').write_text(text,encoding='utf-8')
    # Static HTML is generated from receipts, never embedded model-result claims.
    table=''.join('<tr><th>'+html.escape(s)+'</th>'+''.join('<td>'+f'{r[k]:,}'+'</td>' for k in ['input_pairs','quantitative_observations','structural_observations','distinct_structure_views','components'])+'</tr>' for s,r in clean['split_counts'].items())
    page='<!doctype html><meta charset="utf-8"><title>PepSeek curated dual supervision</title><style>body{max-width:1100px;margin:40px auto;font:16px/1.6 system-ui;color:#20212a}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:right}th:first-child{text-align:left}pre{white-space:pre-wrap;background:#f6f6fa;padding:24px;border-radius:12px}a{color:#5b2b90}</style><h1>PepSeek 两类监督数据集</h1><p>来源、身份、标签、分区和逐条处理状态均可追溯。</p><table><tr><th>分区</th><th>模型配对</th><th>定量观察</th><th>结构观察</th><th>结构视图</th><th>分组</th></tr>'+table+'</table><p><a href="README_ZH.md">完整说明</a> · <a href="audit/CLEAN_RECEIPT.json">规模回执</a> · <a href="audit/SPLIT_RECEIPT.json">划分审计</a> · <a href="audit/ACCEPTANCE.json">验收</a></p><pre>'+html.escape(text)+'</pre>'
    (root/'index.html').write_text(page,encoding='utf-8')
    save(root/'DATA_CARD.json',dict(catalog=cat,clean=clean,split=split,acceptance=accept))
if __name__=='__main__':main()
