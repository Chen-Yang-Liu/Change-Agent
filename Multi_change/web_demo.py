import copy
import os

import streamlit as st
from streamlit.logger import get_logger

from lagent.actions import ActionExecutor, GoogleSearch, PythonInterpreter, Visual_Change_Process_PythonInterpreter
from lagent.agents.react import ReAct
from lagent.llms import GPTAPI
from lagent.llms.huggingface import HFTransformerCasualLM
os.environ["SERPER_API_KEY"] = 'xxxxxxx'
opanai_key = 'xxxxxxxxx'
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
class SessionState:

    def init_state(self):
        """Initialize session state variables."""
        st.session_state['assistant'] = []
        st.session_state['user'] = []
        st.session_state['history'] = []

        action_list = [Visual_Change_Process_PythonInterpreter(), GoogleSearch()]
        st.session_state['plugin_map'] = {
            action.name: action
            for action in action_list
        }
        st.session_state['model_map'] = {}
        st.session_state['model_selected'] = None
        st.session_state['plugin_actions'] = set()

    def clear_state(self):
        """Clear the existing session state."""
        st.session_state['assistant'] = []
        st.session_state['user'] = []
        st.session_state['history'] = []
        st.session_state['model_selected'] = None
        if 'chatbot' in st.session_state:
            st.session_state['chatbot']._session_history = []


class StreamlitUI:

    def __init__(self, session_state: SessionState):
        self.init_streamlit()
        self.session_state = session_state

    def init_streamlit(self):
        """Initialize Streamlit's UI settings."""
        st.set_page_config(
            layout='wide',
            page_title='RSAgent-web',
            page_icon='./docs/imgs/lagent_icon.png')
        st.header('🌏 :blue[RS] Agent ', divider='rainbow')

        st.sidebar.title('Configuration')

    def setup_sidebar(self):
        """Setup the sidebar for model and plugin selection."""
        model_name = st.sidebar.selectbox(
            '**Language Model Selection:**', options=['gpt-3.5-turbo', 'internlm'])
        if model_name != st.session_state['model_selected']:
            model = self.init_model(model_name)
            self.session_state.clear_state()
            st.session_state['model_selected'] = model_name
            if 'chatbot' in st.session_state:
                del st.session_state['chatbot']
        else:
            model = st.session_state['model_map'][model_name]

        plugin_name = st.sidebar.multiselect(
            '**Tool Selection:**',
            options=list(st.session_state['plugin_map'].keys()),
            # default=[list(st.session_state['plugin_map'].keys())[0]],
            default=list(st.session_state['plugin_map'].keys()),
        )

        plugin_action = [
            st.session_state['plugin_map'][name] for name in plugin_name
        ]
        if 'chatbot' in st.session_state:
            st.session_state['chatbot']._action_executor = ActionExecutor(
                actions=plugin_action)
        if st.sidebar.button('**Clear conversation**', key='clear'):
            self.session_state.clear_state()
        uploaded_file_A = st.sidebar.file_uploader(
            '**Upload Image_A:**', type=['png', 'jpg', 'jpeg'])#, 'mp4', 'mp3', 'wav'
        uploaded_file_B = st.sidebar.file_uploader(
            '**Upload Image_B:**', type=['png', 'jpg', 'jpeg'])
        return model_name, model, plugin_action, uploaded_file_A, uploaded_file_B

    def init_model(self, option):
        """Initialize the model based on the selected option."""
        if option not in st.session_state['model_map']:
            if option.startswith('gpt'):
                st.session_state['model_map'][option] = GPTAPI(
                    model_type=option, key=opanai_key)
            else:
                st.session_state['model_map'][option] = HFTransformerCasualLM(
                    'internlm/internlm2_5-7b-chat')
        return st.session_state['model_map'][option]

    def initialize_chatbot(self, model, plugin_action):
        """Initialize the chatbot with the given model and plugin actions."""
        return ReAct(
            llm=model, action_executor=ActionExecutor(actions=plugin_action))

    def render_user(self, prompt: str):
        with st.chat_message('user'):
            st.markdown(prompt)

    def render_assistant(self, agent_return):
        with st.chat_message('assistant'):
            for action in agent_return.actions:
                if (action):
                    self.render_action(action)
            st.markdown(agent_return.response)

    def render_action(self, action):
        with st.expander(action.type, expanded=True):
            st.markdown(
                "<p style='text-align: left;display:flex;'> <span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'>Tool</span><span style='width:14px;text-align:left;display:block;'>:</span><span style='flex:1;'>"  # noqa E501
                + action.type + '</span></p>',
                unsafe_allow_html=True)
            st.markdown(
                "<p style='text-align: left;display:flex;'> <span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'>Thought</span><span style='width:14px;text-align:left;display:block;'>:</span><span style='flex:1;'>"  # noqa E501
                + action.thought + '</span></p>',
                unsafe_allow_html=True)
            if (isinstance(action.args, dict) and 'text' in action.args):
                st.markdown(
                    "<p style='text-align: left;display:flex;'><span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'>Execution Content</span><span style='width:14px;text-align:left;display:block;'>:</span></p>",  # noqa E501
                    unsafe_allow_html=True)
                st.markdown(action.args['text'])
            self.render_action_results(action)

    def render_action_results(self, action):
        """Render the results of action, including text, images, videos, and
        audios."""
        if (isinstance(action.result, dict)):
            st.markdown(
                "<p style='text-align: left;display:flex;'><span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'> 执行结果</span><span style='width:14px;text-align:left;display:block;'>:</span></p>",  # noqa E501
                unsafe_allow_html=True)
            if 'text' in action.result:
                st.markdown(
                    "<p style='text-align: left;'>" + action.result['text'] +
                    '</p>',
                    unsafe_allow_html=True)
            if 'image' in action.result:
                image_path = action.result['image']
                image_data = open(image_path, 'rb').read()
                st.image(image_data, caption='Generated Image')
            if 'video' in action.result:
                video_data = action.result['video']
                video_data = open(video_data, 'rb').read()
                st.video(video_data)
            if 'audio' in action.result:
                audio_data = action.result['audio']
                audio_data = open(audio_data, 'rb').read()
                st.audio(audio_data)


def main():
    logger = get_logger(__name__)
    # Initialize Streamlit UI and setup sidebar
    if 'ui' not in st.session_state:
        session_state = SessionState()
        session_state.init_state()
        st.session_state['ui'] = StreamlitUI(session_state)

    else:
        st.set_page_config(
            layout='wide',
            page_title='RSAgent-web',
            page_icon='./docs/imgs/lagent_icon.png')
        st.header('🌏:blue[RS] Agent ', divider='rainbow')
    model_name, model, plugin_action, uploaded_file_A, uploaded_file_B = st.session_state[
        'ui'].setup_sidebar()

    # Initialize chatbot if it is not already initialized
    # or if the model has changed
    if 'chatbot' not in st.session_state or model != st.session_state[
            'chatbot']._llm:
        st.session_state['chatbot'] = st.session_state[
            'ui'].initialize_chatbot(model, plugin_action)

    for prompt, agent_return in zip(st.session_state['user'],
                                    st.session_state['assistant']):
        st.session_state['ui'].render_user(prompt)
        st.session_state['ui'].render_assistant(agent_return)
    # User input form at the bottom (this part will be at the bottom)
    # with st.form(key='my_form', clear_on_submit=True):

    if user_input := st.chat_input(''):
        st.session_state['ui'].render_user(user_input)
        st.session_state['user'].append(user_input)
        # Add file uploader to sidebar
        if uploaded_file_B:
            file_bytes_B = uploaded_file_B.read()
            file_type_B = uploaded_file_B.type
            if 'image' in file_type_B:
                st.image(file_bytes_B, caption='Uploaded Image_B')#, use_column_width=False, width=300
            # elif 'video' in file_type_B:
            #     st.video(file_bytes_B, caption='Uploaded Video')
            # elif 'audio' in file_type_B:
            #     st.audio(file_bytes_B, caption='Uploaded Audio')
            # Save the file to a temporary location and get the path
            file_path_B = os.path.join(root_dir, uploaded_file_B.name)
            with open(file_path_B, 'wb') as tmpfile:
                tmpfile.write(file_bytes_B)
            st.write(f'File saved at: {file_path_B}')
            user_input = 'The path of the image_B:: {file_path_B}. {user_input}'.format(
                file_path_B=file_path_B, user_input=user_input)
        if uploaded_file_A:
            file_bytes_A = uploaded_file_A.read()
            file_type_A = uploaded_file_A.type
            if 'image' in file_type_A:
                st.image(file_bytes_A, caption='Uploaded Image_A') #, use_column_width=False, width=300
            # elif 'video' in file_type_A:
            #     st.video(file_bytes_A, caption='Uploaded Video')
            # elif 'audio' in file_type_A:
            #     st.audio(file_bytes_A, caption='Uploaded Audio')
            # Save the file to a temporary location and get the path
            file_path_A = os.path.join(root_dir, uploaded_file_A.name)
            with open(file_path_A, 'wb') as tmpfile:
                tmpfile.write(file_bytes_A)
            st.write(f'File saved at: {file_path_A}')
            user_input = 'The path of the image_A: {file_path_A}. {user_input}'.format(
                file_path_A=file_path_A, user_input=user_input)

        print('user_input:', user_input)
        st.session_state['history'].append(dict(role='user', content=user_input))
        agent_return = st.session_state['chatbot'].chat(st.session_state['history'])
        st.session_state['history'].append(dict(role='assistant', content=agent_return.response))
        st.session_state['assistant'].append(copy.deepcopy(agent_return))
        logger.info(agent_return.inner_steps)
        st.session_state['ui'].render_assistant(agent_return)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.join(root_dir, 'tmp_dir')
    os.makedirs(root_dir, exist_ok=True)
    main()
