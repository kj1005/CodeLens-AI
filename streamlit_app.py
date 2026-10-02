import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("CODELENS_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 300

st.set_page_config(page_title="CodeLens AI", page_icon="</>", layout="centered")
st.title("CodeLens AI")
st.caption("Ask grounded questions about a GitHub codebase.")
st.caption(f"API: {API_BASE_URL}")


def response_error(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = {}

    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, list):
        detail = "; ".join(
            str(item.get("msg", item)) if isinstance(item, dict) else str(item)
            for item in detail
        )
    return str(detail or f"The API returned HTTP {response.status_code}.")


def show_request_error(error: requests.RequestException) -> None:
    st.error(f"Could not reach the CodeLens API: {error}")


if "indexed_repository" not in st.session_state:
    st.session_state.indexed_repository = None
if "index_result" not in st.session_state:
    st.session_state.index_result = None

st.subheader("Index a repository")
with st.form("repository_index_form"):
    github_url = st.text_input(
        "GitHub repository URL",
        placeholder="https://github.com/owner/repository",
    )
    index_submitted = st.form_submit_button("Index Repository", type="primary")

if index_submitted:
    if not github_url.strip():
        st.error("Enter a GitHub repository URL.")
    else:
        try:
            response = requests.post(
                f"{API_BASE_URL}/repository/index-to-chroma",
                json={"url": github_url.strip()},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if not response.ok:
                st.error(f"Indexing failed: {response_error(response)}")
                st.session_state.index_result = None
            else:
                result = response.json()
                st.session_state.index_result = result
                st.session_state.indexed_repository = result.get("repository_name")
        except requests.RequestException as error:
            st.session_state.index_result = None
            show_request_error(error)
        except ValueError:
            st.session_state.index_result = None
            st.error("The API returned an invalid indexing response.")

index_result = st.session_state.index_result
if index_result:
    st.success(f"Indexed {index_result.get('repository_name', 'repository')}.")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Files", index_result.get("file_count", 0))
    metric_columns[1].metric("Chunks", index_result.get("total_chunks", 0))
    metric_columns[2].metric("Vectors stored", index_result.get("vectors_stored", 0))

st.divider()
st.subheader("Ask about the codebase")
indexed_repository = st.session_state.indexed_repository
with st.form("repository_question_form"):
    question = st.text_area(
        "Question",
        placeholder="Where are user records loaded?",
        height=100,
    )
    control_columns = st.columns([1, 1])
    with control_columns[0]:
        top_k = st.number_input("Source results", min_value=1, max_value=50, value=5)
    with control_columns[1]:
        explanation_level = st.selectbox(
            "Explanation level",
            ["Beginner", "Intermediate", "Expert"],
            index=1,
        )
    limit_to_indexed_repository = st.checkbox(
        "Limit to indexed repository",
        value=bool(indexed_repository),
        disabled=not bool(indexed_repository),
    )
    ask_submitted = st.form_submit_button("Ask CodeLens", type="primary")

if ask_submitted:
    if not question.strip():
        st.error("Enter a question about the codebase.")
    else:
        request_body = {
            "query": question.strip(),
            "top_k": int(top_k),
            "explanation_level": explanation_level.lower(),
        }
        if limit_to_indexed_repository and indexed_repository:
            request_body["repository_name"] = indexed_repository

        try:
            response = requests.post(
                f"{API_BASE_URL}/repository/ask",
                json=request_body,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if not response.ok:
                st.error(f"Question failed: {response_error(response)}")
            else:
                result = response.json()
                st.markdown("### Answer")
                st.markdown(result.get("answer", "No answer was returned."))

                sources = result.get("results", [])
                st.markdown(f"### Sources ({len(sources)})")
                if not sources:
                    st.info("No source chunks were retrieved for this question.")
                for source in sources:
                    file_path = source.get("file_path", "Unknown file")
                    start_line = source.get("start_line", "?")
                    end_line = source.get("end_line", "?")
                    language = source.get("language", "text").lower()
                    with st.expander(f"{file_path} · Lines {start_line}-{end_line}"):
                        st.code(source.get("document", ""), language=language)
                        score_columns = st.columns(2)
                        score_columns[0].caption(f"RRF score: {source.get('rrf_score')}")
                        score_columns[1].caption(f"Rerank score: {source.get('rerank_score')}")
        except requests.RequestException as error:
            show_request_error(error)
        except ValueError:
            st.error("The API returned an invalid answer response.")