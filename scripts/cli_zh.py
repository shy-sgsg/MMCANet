"""为项目命令行参数解析提供简体中文提示。"""
from __future__ import annotations

import argparse
import sys


class ChineseArgumentParser(argparse.ArgumentParser):
    """中文化 argparse 的用法、分组标题与参数错误。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for group in self._action_groups:
            if group.title == "positional arguments":
                group.title = "位置参数"
            elif group.title == "options":
                group.title = "可选参数"
        for action in self._actions:
            if action.dest == "help":
                action.help = "显示此帮助信息并退出"

    def format_usage(self) -> str:
        usage = super().format_usage()
        if usage.startswith("usage:"):
            return "用法：" + usage[len("usage:") :]
        return usage

    def format_help(self) -> str:
        help_text = super().format_help()
        if help_text.startswith("usage:"):
            help_text = "用法：" + help_text[len("usage:") :]
        return help_text.replace("位置参数:\n", "位置参数：\n").replace("可选参数:\n", "可选参数：\n")

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        if message.startswith("unrecognized arguments:"):
            detail = "无法识别的参数：" + message[len("unrecognized arguments:") :].strip()
        elif message.startswith("the following arguments are required:"):
            detail = "缺少必需参数：" + message[len("the following arguments are required:") :].strip()
        elif "expected one argument" in message:
            argument = message.partition(":")[0].removeprefix("argument ")
            detail = f"参数 {argument} 缺少取值。"
        else:
            detail = "参数无效，请检查参数名称、取值和格式。"
        self.exit(2, f"{self.prog}: 参数错误：{detail}\n")
