from typing import List, Dict, Any, Optional
# 修复 LLMChain 的导入
try:
    # 首先尝试从标准位置导入（在v1.x中可能已不在此处）
    from langchain.chains import LLMChain
except ImportError:
    try:
        # 关键：从 langchain-classic 兼容包中导入
        from langchain_classic.chains import LLMChain
        print("信息: LLMChain 从 langchain_classic.chains 导入 (兼容模式)")
    except ImportError as e:
        # 如果以上全部失败
        raise ImportError(f"无法导入 LLMChain。请确认已安装 'langchain-classic' 包。原始错误: {e}")
from langchain_core.prompts import PromptTemplate
from datetime import datetime
import markdown
import pdfkit
import os
from pathlib import Path
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class ResearchWriter:
    """研究报告生成器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.output_dir = Path(settings.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, 
                       topic: str, 
                       research_data: Dict[str, Any],
                       format: str = "markdown") -> str:
        """生成研究报告"""
        
        template = """# 研究报告：{topic}
        
生成时间：{timestamp}
        
## 摘要
        
{summary}
        
## 研究背景
        
{background}
        
## 关键发现
        
{findings}
        
## 详细分析
        
{analysis}
        
## 结论
        
{conclusion}
        
## 参考文献
        
{references}
        
## 附录：研究过程
        
{process}
        """
        
        report_data = {
            "topic": topic,
            "timestamp": datetime.now().strftime("%Y年%m月%d日 %H:%M:%S"),
            "summary": research_data.get("summary", ""),
            "background": research_data.get("background", ""),
            "findings": self._format_findings(research_data.get("findings", [])),
            "analysis": research_data.get("analysis", ""),
            "conclusion": research_data.get("conclusion", ""),
            "references": self._format_references(research_data.get("references", [])),
            "process": research_data.get("process", "")
        }
        
        report = template.format(**report_data)
        
        filename = self._save_report(topic, report, format)
        
        return filename
    
    def _format_findings(self, findings: List[str]) -> str:
        """格式化关键发现"""
        if not findings:
            return "暂无关键发现"
        
        formatted = ""
        for i, finding in enumerate(findings, 1):
            formatted += f"{i}. {finding}\n"
        
        return formatted
    
    def _format_references(self, references: List[Dict[str, str]]) -> str:
        """格式化参考文献"""
        if not references:
            return "暂无参考文献"
        
        formatted = ""
        for i, ref in enumerate(references, 1):
            formatted += f"{i}. {ref.get('title', '无标题')}\n"
            if ref.get('author'):
                formatted += f"   作者: {ref['author']}\n"
            if ref.get('source'):
                formatted += f"   来源: {ref['source']}\n"
            if ref.get('date'):
                formatted += f"   日期: {ref['date']}\n"
            formatted += "\n"
        
        return formatted
    
    def _save_report(self, topic: str, report: str, format: str) -> str:
        """保存报告到文件"""
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_topic = safe_topic.replace(' ', '_')[:50]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "markdown":
            filename = self.output_dir / f"{safe_topic}_{timestamp}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Markdown报告已保存: {filename}")
            
            html_filename = self.output_dir / f"{safe_topic}_{timestamp}.html"
            html_content = markdown.markdown(report, extensions=['tables', 'fenced_code'])
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>研究报告: {topic}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
                        h1 {{ color: #333; border-bottom: 2px solid #eee; }}
                        h2 {{ color: #555; }}
                        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
                        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                    </style>
                </head>
                <body>
                    {html_content}
                </body>
                </html>
                """)
            logger.info(f"HTML报告已保存: {html_filename}")
            
        elif format.lower() == "pdf":
            filename = self.output_dir / f"{safe_topic}_{timestamp}.pdf"
            try:
                # 关键配置：指定 wkhtmltopdf 的绝对路径
                pdfkit_config = pdfkit.configuration(
                    wkhtmltopdf=r"F:\wkhtmltopdf_Folder\wkhtmltopdf_Application\wkhtmltopdf\bin\wkhtmltopdf.exe"
                )
                pdfkit.from_string(report, str(filename), configuration=pdfkit_config)
                logger.info(f"PDF报告已保存: {filename}")
            except Exception as e:
                logger.error(f"生成PDF时出错: {str(e)}")
                # 回退到生成Markdown格式
                filename = self._save_report(topic, report, "markdown")
        else:
            filename = self.output_dir / f"{safe_topic}_{timestamp}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"文本报告已保存: {filename}")
        
        return str(filename)
    
    def generate_summary_chain(self) -> LLMChain:
        """创建摘要生成链"""
        prompt = PromptTemplate(
            input_variables=["content"],
            template="请为以下内容生成一个简洁的摘要：\n\n{content}\n\n摘要："
        )
        return LLMChain(llm=self.llm, prompt=prompt)
