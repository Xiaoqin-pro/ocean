"""Deterministic component, box-merging, and GT-oracle crop primitives for R1a."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class Box:
    x0: int; y0: int; x1: int; y1: int
    def expand(self, padding: int, width: int, height: int) -> "Box": return Box(max(0,self.x0-padding),max(0,self.y0-padding),min(width,self.x1+padding),min(height,self.y1+padding))
    def touches(self, other: "Box") -> bool: return self.x0 <= other.x1 and other.x0 <= self.x1 and self.y0 <= other.y1 and other.y0 <= self.y1
    def union(self, other: "Box") -> "Box": return Box(min(self.x0,other.x0),min(self.y0,other.y0),max(self.x1,other.x1),max(self.y1,other.y1))
    @property
    def area(self) -> int: return (self.x1-self.x0)*(self.y1-self.y0)

def error_component_boxes(error: np.ndarray) -> list[Box]:
    count, labels = cv2.connectedComponents(np.asarray(error, dtype=np.uint8), connectivity=8)
    boxes=[]
    for ident in range(1,count):
        ys,xs=np.where(labels==ident)
        boxes.append(Box(int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)))
    return sorted(boxes,key=lambda b:(b.y0,b.x0,b.y1,b.x1))

def merged_context_groups(core: list[Box], width: int, height: int, padding: int=32) -> list[tuple[Box,list[Box]]]:
    contexts=[box.expand(padding,width,height) for box in core]; groups=[[i] for i in range(len(core))]
    changed=True
    while changed:
        changed=False
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                a=contexts[groups[i][0]]
                for index in groups[i][1:]: a=a.union(contexts[index])
                b=contexts[groups[j][0]]
                for index in groups[j][1:]: b=b.union(contexts[index])
                if a.touches(b): groups[i]+=groups[j]; del groups[j]; changed=True; break
            if changed: break
    result=[]
    for indices in groups:
        context=contexts[indices[0]]
        for idx in indices[1:]: context=context.union(contexts[idx])
        result.append((context,[core[idx] for idx in indices]))
    return sorted(result,key=lambda item:(item[0].y0,item[0].x0))

def core_mask(groups: list[tuple[Box,list[Box]]], height: int, width: int) -> np.ndarray:
    mask=np.zeros((height,width),dtype=bool)
    for _,boxes in groups:
        for box in boxes: mask[box.y0:box.y1,box.x0:box.x1]=True
    return mask
