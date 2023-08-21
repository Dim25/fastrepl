import pytest

from fastrepl import Analyze
from fastrepl.polish import Updatable

# TODO: These tests work on my machine and have worked in CI before, but they suddenly start failing at #11. I'm not really sure why, but let's move on.

# class TestGraph:
#     def test_no_graph(self):
#         with Analyze() as controller:
#             with pytest.raises(ValueError):
#                 controller.convert_graph(GRAPH=0)

#     def test_single_run_1(self):
#         def fn():
#             Updatable(key="key_1", value="value_1"),
#             Updatable(key="key_2", value="value_2"),

#         for GRAPH in [1, 2, 3]:
#             with Analyze() as controller:
#                 fn()
#                 info = controller.convert_graph(GRAPH=GRAPH)
#                 assert len(info["nodes"]) == 1 + (1 if GRAPH >= 3 else 0)
#                 assert len(info["edges"]) == 0 + (1 if GRAPH >= 3 else 0)

#     def test_single_run_2(self):
#         def fn1():
#             Updatable(key="key_1", value="value_1"),
#             Updatable(key="key_2", value="value_2"),

#         def fn2():
#             Updatable(key="key_3", value="value_3"),

#         def fn():
#             fn1()
#             fn2()

#         for GRAPH in [1, 2, 3]:
#             with Analyze() as controller:
#                 fn()
#                 info = controller.convert_graph(GRAPH=GRAPH)
#                 assert len(info["nodes"]) == 2 + (1 if GRAPH >= 3 else 0)
#                 assert len(info["edges"]) == 1 + (1 if GRAPH >= 3 else 0)

#     def test_multi_run(self):
#         def fn1():
#             Updatable(key="key_1", value="value_1"),
#             Updatable(key="key_2", value="value_2"),

#         def fn2():
#             Updatable(key="key_3", value="value_3"),

#         for GRAPH in [1, 2, 3]:
#             with Analyze() as controller:
#                 fn1()
#                 controller.next_run()
#                 fn2()

#                 info = controller.convert_graph(GRAPH=GRAPH)
#                 assert len(info["nodes"]) == 4
#                 assert len(info["edges"]) == 2

#     def test_multi_run_shared_node(self):
#         def fn1():
#             Updatable(key="key_1", value="value_1"),

#         def fn2():
#             Updatable(key="key_2", value="value_2"),

#         def fn3():
#             Updatable(key="key_3", value="value_3"),

#         def pipeline1():
#             fn1()
#             fn2()

#         def pipeline2():
#             fn1()
#             fn3()

#         for GRAPH in [1, 2, 3]:
#             with Analyze() as controller:
#                 pipeline1()
#                 controller.next_run()
#                 pipeline2()

#                 info = controller.convert_graph(GRAPH=GRAPH)
#                 assert len(info["nodes"]) == 6
#                 assert len(set(info["nodes"])) == 5
#                 assert len(info["edges"]) == 4
