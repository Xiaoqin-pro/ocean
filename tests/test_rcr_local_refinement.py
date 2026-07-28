import numpy as np
from reliability.rcr_local_refinement import Box, core_mask, error_component_boxes, merged_context_groups
def test_components_boxes_and_transitive_touch_merge_are_deterministic():
    e=np.zeros((10,20),bool); e[1,1]=1; e[1,6]=1; e[1,11]=1
    core=error_component_boxes(e); assert core==[Box(1,1,2,2),Box(6,1,7,2),Box(11,1,12,2)]
    groups=merged_context_groups(core,20,10,padding=4); assert len(groups)==1 and groups[0][0]==Box(0,0,16,6)
    assert core_mask(groups,10,20).sum()==3
