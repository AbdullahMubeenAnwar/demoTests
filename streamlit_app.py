############ IMPORTING LIBRARIES ############

# Import streamlit, requests for API calls, and pandas and numpy for data manipulation

import streamlit as st

############ SETTING UP THE PAGE LAYOUT AND TITLE ############

st.set_page_config(
    layout="wide", page_title="Text Summarization", page_icon="images/jsl_logo_icon.png", initial_sidebar_state="auto"
)

############ HEADING ############

st.caption("")
st.title("Summarize Text")

# We need to set up session state via st.session_state so that app interactions don't reset the app.

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

############ SIDEBAR CONTENT ############

st.sidebar.write("")

logo = st.sidebar.image("images/logo.png", width=300)
model = st.sidebar.selectbox("Choose the pretrained model", ['t5_base', 't5_small'], help="For more info about the models visit: https://sparknlp.org/models",)

# Let's add the colab link for the notebook.

try_link="""<a href="https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5TRANSFORMER.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Try it yourself:')
st.sidebar.markdown(try_link, unsafe_allow_html=True)


############ MAIN CONTENT ############

text_mapping_dict = {

    "Mount Tai is a mountain of historical and cultural significance located north of the city of Tai'an, in Shandong province, China. The tallest peak is the Jade Emperor Peak, which is commonly reported as being 1,545 meters tall, but is officially described by the PRC government as 1,532.7 meters tall. It is associated with sunrise, birth, and renewal, and is often regarded the foremost of the five. Mount Tai has been a place of worship for at least 3,000 years and served as one of the most important ceremonial centers of China during large portions of this period.":
    {
        "t5_base": "the jade Emperor peak is 1,532.7 meters tall . it is associated with sunrise, birth, and renewal .",
        "t5_small": "the tallest peak is the Jade Emperor Peak, which is commonly reported as being 1,545 meters tall . it is associated with sunrise, birth, and renewal ."
    },
    "The Guadeloupe amazon (Amazona violacea) is a hypothetical extinct species of parrot that is thought to have been endemic to the Lesser Antillean island region of Guadeloupe. Described by 17th- and 18th-century writers, it is thought to have been related to, or possibly the same as, the extant imperial amazon. A tibiotarsus and an ulna bone from the island of Marie-Galante may belong to the Guadeloupe amazon. According to contemporary descriptions, its head, neck and underparts were mainly violet or slate, mixed with green and black; the back was brownish green; and the wings were green, yellow and red. It had iridescent feathers, and was able to raise a \"ruff\" of feathers around its neck. It fed on fruits and nuts, and the male and female took turns sitting on the nest. French settlers ate the birds and destroyed their habitat. Rare by 1779, the species appears to have become extinct by the end of the 18th century.":
    {
        "t5_base": "the parrot is thought to have been endemic to the Lesser Antillean island region of Guadeloupe . it is thought to have been related to, or possibly the same as, the extant imperial amazon . rare by 1779, the species appears to have become extinct by the end of the 18th century.",
        "t5_small": "the species of parrot is thought to have been endemic to the Lesser Antillean island region of Guadeloupe . it is thought to have been related to, or possibly the same as, the extant imperial amazon ."
    },
    "Pierre-Simon, marquis de Laplace (23 March 1749 – 5 March 1827) was a French scholar and polymath whose work was important to the development of engineering, mathematics, statistics, physics, astronomy, and philosophy. He summarized and extended the work of his predecessors in his five-volume Mécanique Céleste (Celestial Mechanics) (1799–1825). This work translated the geometric study of classical mechanics to one based on calculus, opening up a broader range of problems. In statistics, the Bayesian interpretation of probability was developed mainly by Laplace.":
        {
        "t5_base": "Pierre-Simon, marquis de Laplace (23 March 1749 – 5 March 1827) was a french scholar and polymath . his work was important to the development of engineering, mathematics, statistics, physics, astronomy, and philosophy ",
        "t5_small": "marquis de Laplace was a polymath and scholar of engineering, mathematics, statistics, physics, astronomy, and philosophy . he compiled and extended the work of his predecessors in his five-volume Mécanique Céleste (1799–1825)"
    },
    "John Snow (15 March 1813 – 16 June 1858) was an English physician and a leader in the development of anaesthesia and medical hygiene. He is considered one of the founders of modern epidemiology, in part because of his work in tracing the source of a cholera outbreak in Soho, London, in 1854, which he curtailed by removing the handle of a water pump. Snow's findings inspired the adoption of anaesthesia as well as fundamental changes in the water and waste systems of London, which led to similar changes in other cities, and a significant improvement in general public health around the world.":
        {
        "t5_base": "he is considered one of the founders of modern epidemiology . his work in tracing the source of a cholera outbreak in soho, London, in 1854 . his findings inspired the adoption of anaesthesia and fundamental changes in the water and waste systems of London .",
        "t5_small": "he was an English physician and a leader in the development of anaesthesia and medical hygiene . he is considered one of the founders of modern epidemiology . his findings inspired the adoption of anaesthesia and fundamental changes in the water and waste systems of London ."
    },
    "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci. Considered an archetypal masterpiece of the Italian Renaissance, it has been described as \"the best known, the most visited, the most written about, the most sung about, the most parodied work of art in the world\". The painting's novel qualities include the subject's enigmatic expression, the monumentality of the composition, the subtle modelling of forms, and the atmospheric illusionism.":
        {
        "t5_base": "the mona Lisa is a half-length portrait painting by italian artist Leonardo da Vinci . it has been described as \"the most parodied work of art in the world\"",
        "t5_small": "the painting is a half-length portrait painting by italian artist Leonardo da Vinci . it is considered an archetypal masterpiece of the italian Renaissance ."
    },
    """Calculus, originally called infinitesimal calculus or "the calculus of infinitesimals", is the mathematical study of continuous change, in the same way that geometry is the study of shape and algebra is the study of generalizations of arithmetic operations. It has two major branches, differential calculus and integral calculus; the former concerns instantaneous rates of change, and the slopes of curves, while integral calculus concerns accumulation of quantities, and areas under or between curves. These two branches are related to each other by the fundamental theorem of calculus, and they make use of the fundamental notions of convergence of infinite sequences and infinite series to a well-defined limit.[1] Infinitesimal calculus was developed independently in the late 17th century by Isaac Newton and Gottfried Wilhelm Leibniz.[2][3] Today, calculus has widespread uses in science, engineering, and economics.[4] In mathematics education, calculus denotes courses of elementary mathematical analysis, which are mainly devoted to the study of functions and limits. The word calculus (plural calculi) is a Latin word, meaning originally "small pebble" (this meaning is kept in medicine – see Calculus (medicine)). Because such pebbles were used for calculation, the meaning of the word has evolved and today usually means a method of computation. It is therefore used for naming specific methods of calculation and related theories, such as propositional calculus, Ricci calculus, calculus of variations, lambda calculus, and process calculus.""":
    {
        "t5_base": """the term calculus is a Latin word, meaning originally "small pebble" it is used for naming specific methods of calculation and related theories .""",
        "t5_small": "calculus is the mathematical study of continuous change . it has two major branches, differential calculus and integral calculus . the latter concerns instantaneous rates of change, and slopes of curves . integral calculus is the study of generalizations of arithmetic operations ."
        }
}

st.subheader("Summarize text to make it shorter while retaining meaning.")

selected_text = st.selectbox("Select an example", list(text_mapping_dict.keys()))

st.subheader('Text')
st.write(selected_text)

st.subheader("Summary")

if model == "t5_base":
    st.write(text_mapping_dict[selected_text]["t5_base"])
else:
    st.write(text_mapping_dict[selected_text]['t5_small'])
