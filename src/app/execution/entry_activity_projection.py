"""Incremental, memory-only index of the existing live-order journal."""
from __future__ import annotations

from bisect import bisect_right, insort
from itertools import islice
import json
import math
import time


class HistoryUnavailable(ValueError):
    pass


class EntryActivityProjection:
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.files = {}
        self.nodes = []
        self.bindings = {}
        self.children = {}
        self.fills = []
        self.fills_by_id = {}
        self.relevant_times = []
        self.earliest = None
        self.had_rotations = False
        self.started = False
        self.sequence = 0
        self.roots = {}
        self.failure = None

    def update(self, paths, signature, *, maximum_bytes, maximum_records, deadline):
        """Resume the bootstrap or consume only appended bytes after bootstrap.

        Offsets follow file identities across numbered rotations. Work budgets
        apply per call, so a journal larger than one budget eventually completes.
        Neither a source file nor an operational database is modified.
        """
        if self.failure:
            raise HistoryUnavailable(self.failure)
        if not self.started:
            self.had_rotations = len(paths) > 1
            self.started = True
        present = {row[1] for row in signature}
        for identity, cursor in self.files.items():
            if identity not in present and (cursor["offset"] < cursor["size"] or cursor["pending"]):
                raise HistoryUnavailable("ENTRY_ACTIVITY_ROTATION_HISTORY_INCOMPLETE")
        for metadata in signature:
            cursor = self.files.setdefault(metadata[1], {
                "offset": 0, "pending": b"", "size": metadata[2], "mtime": metadata[3],
            })
            if (metadata[2] < cursor["size"] or
                    (metadata[2] == cursor["size"] and metadata[3] != cursor["mtime"] and cursor["offset"] > 0)):
                raise HistoryUnavailable("ENTRY_ACTIVITY_HISTORY_TRUNCATED")
            cursor["size"], cursor["mtime"] = metadata[2], metadata[3]
        read_bytes = records = 0
        for path, metadata in reversed(tuple(zip(paths, signature))):
            identity = metadata[1]
            size = metadata[2]
            cursor = self.files[identity]
            if size < cursor["offset"]:
                raise HistoryUnavailable("ENTRY_ACTIVITY_HISTORY_TRUNCATED")
            with path.open("rb") as handle:
                while True:
                    if time.monotonic() > deadline:
                        return False
                    pending = cursor["pending"]
                    newline = pending.find(b"\n")
                    if newline >= 0:
                        if records >= maximum_records:
                            return False
                        line = pending[:newline]
                        cursor["pending"] = pending[newline + 1:]
                        if line:
                            try:
                                self._ingest(json.loads(line))
                            except (ValueError, TypeError, UnicodeError) as exc:
                                self.failure = (str(exc) if isinstance(exc, HistoryUnavailable)
                                                else "ENTRY_ACTIVITY_HISTORY_UNREADABLE")
                                raise HistoryUnavailable(self.failure) from None
                            records += 1
                        continue
                    if cursor["offset"] == size:
                        if pending:
                            raise HistoryUnavailable("ENTRY_ACTIVITY_TRUNCATED_RECORD")
                        break
                    if read_bytes >= maximum_bytes or records >= maximum_records:
                        return False
                    amount = min(65536, size - cursor["offset"], maximum_bytes - read_bytes)
                    handle.seek(cursor["offset"])
                    block = handle.read(amount)
                    if not block:
                        raise HistoryUnavailable("ENTRY_ACTIVITY_HISTORY_CHANGED_DURING_READ")
                    cursor["offset"] += len(block)
                    cursor["pending"] += block
                    read_bytes += len(block)
                    if len(cursor["pending"]) > 2 * 1024 * 1024:
                        raise HistoryUnavailable("ENTRY_ACTIVITY_RECORD_TOO_LARGE")
        return True

    def _ingest(self, event):
        if not isinstance(event, dict) or not isinstance(event.get("payload"), dict):
            raise HistoryUnavailable("ENTRY_ACTIVITY_RECORD_INVALID")
        recorded = self.timestamp(event.get("recorded_at"))
        self.earliest = recorded if self.earliest is None else min(self.earliest, recorded)
        payload = event["payload"]
        kind = event.get("event_type")
        self.sequence += 1
        sequence = self.sequence
        if kind == "live_order_status":
            raw = payload.get("raw")
            if not isinstance(raw, dict):
                raise HistoryUnavailable("ENTRY_ACTIVITY_STATUS_INVALID")
            try:
                quantity = float(raw.get("quantity", 0))
            except (ValueError, TypeError, OverflowError):
                raise HistoryUnavailable("ENTRY_ACTIVITY_STATUS_INVALID") from None
            if not math.isfinite(quantity) or quantity < 0 or not quantity.is_integer():
                raise HistoryUnavailable("ENTRY_ACTIVITY_STATUS_INVALID")
            side = str(raw.get("side", "")).upper()
            if quantity > 0 and side not in {"BUY", "SELL"}:
                raise HistoryUnavailable("ENTRY_ACTIVITY_STATUS_INVALID")
            if quantity <= 0 or side != "BUY":
                return
            order_id = str(payload.get("order_id") or "")
            observed = self.timestamp(payload.get("observed_at"))
            if (not order_id or observed > recorded or
                    (raw.get("order_id") is not None and str(raw["order_id"]) != order_id)):
                raise HistoryUnavailable("ENTRY_ACTIVITY_STATUS_INVALID")
            item = (observed, sequence, order_id, recorded, str(raw.get("ticker") or "").upper())
            insort(self.fills, item)
            insort(self.fills_by_id.setdefault(order_id, []), item)
        elif kind in {"live_order_submitted", "live_order_amended"}:
            order_id = str(payload.get("broker_order_id") or "")
            if not order_id:
                return
            node = {"at": recorded, "sequence": sequence, "id": order_id,
                    "previous": str(payload.get("previous_broker_order_id") or ""),
                    "origin": kind == "live_order_submitted", "contract": _contract(payload.get("order"))}
            index = len(self.nodes)
            self.nodes.append(node)
            insort(self.bindings.setdefault(order_id, []), (recorded, sequence, index))
            if not node["origin"] and node["previous"]:
                self.children.setdefault(node["previous"], set()).add(order_id)
            self.roots.clear()  # A delayed earlier record can complete ancestry.
        else:
            return
        insort(self.relevant_times, recorded)

    def _binding(self, order_id, at, sequence=math.inf, *, ticker=None, contract=None):
        history = self.bindings.get(order_id, ())
        offset = bisect_right(history, (at, sequence, math.inf)) - 1
        while offset >= 0:
            index = history[offset][2]
            terms = self.nodes[index]["contract"]
            # Broker order IDs need not be globally unique across markets.
            # A malformed matching scope remains an error, not a fallback grant.
            if terms is None or ((ticker is None or terms[0] == ticker)
                                 and (contract is None or terms == contract)):
                return index
            offset -= 1
        return None

    def _root(self, index, deadline):
        chain = []
        current = index
        while current is not None and current not in self.roots:
            if time.monotonic() > deadline:
                raise HistoryUnavailable("ENTRY_ACTIVITY_READ_BUDGET_EXCEEDED")
            if current in chain:
                raise HistoryUnavailable("ENTRY_ACTIVITY_AMENDMENT_CYCLE")
            chain.append(current)
            node = self.nodes[current]
            if node["origin"]:
                self.roots[current] = current
                break
            current = self._binding(node["previous"], node["at"], node["sequence"] - 1,
                                    contract=node["contract"])
        root = self.roots.get(current) if current is not None else None
        if root is None:
            raise HistoryUnavailable("ENTRY_ACTIVITY_HISTORY_INCOMPLETE")
        terms = self.nodes[root]["contract"]
        if terms is None or any(self.nodes[item]["contract"] != terms for item in chain):
            raise HistoryUnavailable("ENTRY_ACTIVITY_ORDER_ID_CONFLICT")
        for item in chain:
            self.roots[item] = root
        return root

    def _fill_root(self, fill, deadline):
        observed, _, order_id, _, ticker = fill
        index = self._binding(order_id, observed, ticker=ticker)
        if index is None:
            raise HistoryUnavailable("ENTRY_ACTIVITY_HISTORY_INCOMPLETE")
        root = self._root(index, deadline)
        terms = self.nodes[root]["contract"]
        if terms is None or terms[0] != ticker or terms[2] != "BUY":
            raise HistoryUnavailable("ENTRY_ACTIVITY_ORDER_ID_CONFLICT")
        return root

    def count(self, market, day_start, now, *, deadline):
        if self.had_rotations and (self.earliest is None or self.earliest >= day_start):
            raise HistoryUnavailable("ENTRY_ACTIVITY_HISTORY_INCOMPLETE")
        start = bisect_right(self.fills, (day_start, -math.inf))
        stop = bisect_right(self.fills, (now, math.inf))
        candidates = set()
        for fill in islice(self.fills, start, stop):
            if time.monotonic() > deadline:
                raise HistoryUnavailable("ENTRY_ACTIVITY_READ_BUDGET_EXCEEDED")
            if fill[3] <= now:
                root = self._fill_root(fill, deadline)
                if self.nodes[root]["contract"][1:] == (market, "BUY", "LONG", "OPEN", "CASH"):
                    candidates.add(root)
        count = 0
        for root in candidates:
            node = self.nodes[root]
            first = None
            pending, seen = [node["id"]], set()
            while pending:
                if time.monotonic() > deadline:
                    raise HistoryUnavailable("ENTRY_ACTIVITY_READ_BUDGET_EXCEEDED")
                order_id = pending.pop()
                if order_id in seen:
                    continue
                seen.add(order_id)
                pending.extend(self.children.get(order_id, ()))
                history = self.fills_by_id.get(order_id, ())
                offset = bisect_right(history, (node["at"], -math.inf))
                for fill in islice(history, offset, None):
                    if time.monotonic() > deadline:
                        raise HistoryUnavailable("ENTRY_ACTIVITY_READ_BUDGET_EXCEEDED")
                    if fill[0] > now:
                        break
                    if fill[3] > now:
                        continue
                    if fill[4] != node["contract"][0]:
                        continue
                    # Same broker IDs can be reused on another day. Only fills
                    # belonging to this economic root determine its first day.
                    if self._fill_root(fill, deadline) == root:
                        first = fill[0] if first is None else min(first, fill[0])
                        break
            if first is not None and first >= day_start:
                count += 1
        next_offset = bisect_right(self.relevant_times, now)
        next_future = self.relevant_times[next_offset] if next_offset < len(self.relevant_times) else None
        return count, next_future


def _contract(order):
    from app.data.market_capabilities import normalize_market_group
    if not isinstance(order, dict):
        return None
    group = normalize_market_group(str(order.get("market") or ""))
    ticker = str(order.get("ticker") or "").upper()
    if group is None or not ticker:
        return None
    return (ticker, group.value, str(order.get("side", "")).upper(),
            str(order.get("position_direction", "LONG")).upper(),
            str(order.get("position_effect") or "OPEN").upper(),
            str(order.get("execution_product", "CASH")).upper())
