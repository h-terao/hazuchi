"""
Modify from:
https://github.com/chainer/chainer/blob/master/chainer/training/extensions/print_report.py
https://github.com/pfnet/pytorch-pfn-extras/blob/master/pytorch_pfn_extras/training/extensions/print_report.py
"""
from __future__ import annotations
from copy import deepcopy
import json

from rich import pretty

from . import callback


# Copy from pytorch-pfn-extras
def create_header_and_templates(
    entries: list[str],
) -> tuple[str, list[tuple[str, str, str]]]:
    """Construct header and templates from `entries`
    Args:
        entries (list): list of str
    Returns:
        header (str): header string
        templates (str): template string for print values.
    """
    # format information
    entry_widths = [max(10, len(s)) for s in entries]

    header = "  ".join(("{:%d}" % w for w in entry_widths)).format(*entries)
    templates = []
    for entry, w in zip(entries, entry_widths):
        templates.append((entry, "{:<%dg}  " % w, " " * (w + 2)))
    return header, templates


# Copy from pytorch-pfn-extras
def filter_and_sort_entries(
    all_entries: list[str],
) -> list[str]:
    entries = deepcopy(all_entries)
    # TODO(nakago): sort other entries if necessary

    if "step" in entries:
        # move iteration to head
        entries.pop(entries.index("step"))
        entries = ["step"] + entries
    if "epoch" in entries:
        # move epoch to head
        entries.pop(entries.index("epoch"))
        entries = ["epoch"] + entries

    return entries


class PrintMetrics(callback.Callback):
    """Print the specified metrics on the console.

    Args:
        entries (list of str): List of metric names to print on the console.
    """

    def __init__(self, entries: list[str]):
        header, templates = create_header_and_templates(entries)
        self._header = header
        self._templates = templates

        self._log = []
        self._log_len = 0

    def on_fit_epoch_end(self, trainer, train_state, summary):
        self._log.append(summary)

        if self._header is not None:
            print(self._header)
            self._header = None

        while len(self._log) > self._log_len:
            s = ""
            for entry, template, empty in self._templates:
                if entry in self._log[self._log_len]:
                    s += template.format(self._log[self._log_len][entry])
                else:
                    s += empty
            print(s)
            self._log_len += 1

        return train_state, summary

    def on_test_epoch_end(self, trainer, train_state, summary):
        pretty.pprint(summary)
        return train_state, summary

    def to_state_dict(self):
        return {"_log": json.dumps(self._log)}

    def from_state_dict(self, state) -> None:
        self._log = json.loads(state["_log"])
