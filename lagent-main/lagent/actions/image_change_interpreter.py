# flake8: noqa: E501
import copy
import io
from contextlib import redirect_stdout
from typing import Any, Optional, Type

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode


class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(
            self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)


class Visual_Change_Process_PythonInterpreter(BaseAction):
    """A Python executor that can execute Python scripts.

    Args:
        answer_symbol (str, Optional): the answer symbol from LLM. Defaults to ``None``.
        answer_expr (str, Optional): the answer function name of the Python
            script. Defaults to ``'solution()'``.
        answer_from_stdout (boolean, Optional): whether the execution results is from
            stdout. Defaults to ``False``.
        timeout (int, Optional): Upper bound of waiting time for Python script execution.
            Defaults to ``20``.
        description (dict, Optional): The description of the action. Defaults to ``None``.
        parser (Type[BaseParser]): The parser class to process the
            action's inputs and outputs. Defaults to :class:`JsonParser`.
        enable (bool, optional): Whether the action is enabled. Defaults to
            ``True``.
    """

    def __init__(self,
                 answer_symbol: Optional[str] = None,
                 answer_expr: Optional[str] = 'solution()',
                 answer_from_stdout: bool = False,
                 timeout: int = 500,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True) -> None:
        super().__init__(description, parser, enable)
        self.answer_symbol = answer_symbol
        self.answer_expr = answer_expr
        self.answer_from_stdout = answer_from_stdout
        self.timeout = timeout

    @tool_api
    def run(self, command: str) -> ActionReturn:
        """用来执行Python代码。代码必须是一个函数，函数名必须得是 'solution'，代码对应你的思考过程。代码实例格式如下：

        ```python
        # import 依赖包
        import xxx
        def solution():
            # 初始化一些变量
            variable_names_with_real_meaning = xxx
            # 步骤一
            mid_variable = func(variable_names_with_real_meaning)
            # 步骤 x
            mid_variable = func(mid_variable)
            # 最后结果
            final_answer =  func(mid_variable)
            return final_answer
        ```
        Note:
        In addition to the commonly used python dependency packages, you can use a custom library 'Change_Perception' from tools import Change_Perception(), which contains the following function:
        1. **`change_detection(path_A, path_B, savepath_mask)`**:
           - **Parameters**:
             - `path_A`: Path to the first image.
             - `path_B`: Path to the second image.
             - `savepath_mask`: Path to save the mask image.
           - **Returns**:
             - A mask map representing the changed areas. The mask is a numpy array with dimensions (256,256), where each pixel value represents the following:
               - 0 stands for unchanged,
               - 1 stands for changed road, and
               - 2 stands for changed building.

        2. **`generate_change_caption(path_A, path_B)`**:
           - **Parameters**:
             - `path_A`: Path to the first image.
             - `path_B`: Path to the second image.
           - **Returns**:
             - A String: Sentences describing the changes between the two images.

        3. **`compute_object_num(change_mask,object)`**:
           - **Parameters**:
             - `change_mask`: The mask from the `change_detection` function.
             - `object`: The object type to be counted. It can be either 'building' or 'road'.
           - **Returns**:
             - The number of changed objects.

        NOTE: The code of Action Input must be placed in def solution()!!
        For example:
        When the user wants to detect the changed buildings and save the changed building areas in red, "Action Input" should be as follows:
        ``python
        def solution():
            from tools import Change_Perception
            import cv2
            import numpy as np
            path_A = 'xxxxxx'
            path_B = 'xxxxxx'
            savepath_mask = 'xxxxxx'
            # initiate Change_Perception
            Change_Perception_model = Change_Perception()
            mask = Change_Perception_model.change_detection(path_A, path_B, savepath_mask)
            mask_bgr = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            mask_bgr[mask == 2] = [0, 0, 255] # '2' stands for changed building (red)
            cv2.imwrite(savepath_mask, mask_bgr)
        ```



        Args:
            command (:class:`str`): Python code snippet
        """
        from func_timeout import FunctionTimedOut, func_set_timeout
        self.runtime = GenericRuntime()
        try:
            tool_return = func_set_timeout(self.timeout)(self._call)(command)
        except FunctionTimedOut as e:
            tool_return = ActionReturn(type=self.name)
            tool_return.errmsg = repr(e)
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return

    def _call(self, command: str) -> ActionReturn:
        tool_return = ActionReturn(type=self.name)
        print('RUN Command:', command)
        try:
            if '```python' in command:
                command = command.split('```python')[1].split('```')[0]
            elif '```' in command:
                command = command.split('```')[1].split('```')[0]
            tool_return.args = dict(text='```python\n' + command + '\n```')
            command = command.split('\n')

            if self.answer_from_stdout:
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    self.runtime.exec_code('\n'.join(command))
                program_io.seek(0)
                res = program_io.readlines()[-1]
            elif self.answer_symbol:
                self.runtime.exec_code('\n'.join(command))
                res = self.runtime._global_vars[self.answer_symbol]
            elif self.answer_expr:
                self.runtime.exec_code('\n'.join(command))
                res = self.runtime.eval_code(self.answer_expr)
            else:
                self.runtime.exec_code('\n'.join(command[:-1]))
                res = self.runtime.eval_code(command[-1])
        except Exception as e:
            print('Model RUN Error:', e)
            tool_return.errmsg = repr(e)
            tool_return.type = self.name
            tool_return.state = ActionStatusCode.API_ERROR
            return tool_return
        try:
            tool_return.result = [dict(type='text', content=str(res))]
            tool_return.state = ActionStatusCode.SUCCESS
        except Exception as e:
            tool_return.errmsg = repr(e)
            tool_return.type = self.name
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return
