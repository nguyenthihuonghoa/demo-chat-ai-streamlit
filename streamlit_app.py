import os
import time
import json
import logging
from typing import Any, Dict, Iterable, List, Tuple, TypedDict, Optional

import requests
import streamlit as st
import base64

logger = logging.getLogger(__name__)


class ChatMessage(TypedDict):
    role: str  # "user" | "assistant"
    content: str

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

BACKEND_URL_DEFAULT: str = os.getenv("HISTORYBOT_BACKEND_URL")


def get_backend_url() -> str:
    url: str = st.session_state.get("backend_url", BACKEND_URL_DEFAULT)
    return url.rstrip("/")


def check_backend_health(backend_url: str, timeout: int = 5) -> tuple[bool, str]:
    try:
        resp = requests.get(f"{backend_url}/", timeout=timeout)
        resp.raise_for_status()
        return True, "✅ Backend đang hoạt động"
    except requests.exceptions.ConnectionError:
        return False, "❌ Không thể kết nối đến backend. Vui lòng kiểm tra:\n- Backend đã chạy chưa?\n- URL có đúng không?"
    except requests.exceptions.Timeout:
        return False, "⏱️ Backend không phản hồi (timeout). Có thể backend đang quá tải."
    except requests.exceptions.RequestException as e:
        return False, f"❌ Lỗi kết nối: {str(e)}"


def parse_ndjson_stream(
    raw_chunks: Iterable[str],
) -> Iterable[Dict[str, Any]]:
    """
    Parse NDJSON stream từ backend.
    Mỗi dòng là một JSON object, nhưng chunks có thể không align với dòng.
    Trả về các message đã parse: {"type": "...", ...}
    """
    buffer: str = ""
    
    for chunk in raw_chunks:
        if not chunk:
            continue
        
        buffer += chunk
        
        # Tách các dòng hoàn chỉnh (có \n)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            
            if not line:
                continue
            
            try:
                message = json.loads(line)
                if isinstance(message, dict):
                    yield message
            except json.JSONDecodeError:
                # Bỏ qua dòng JSON không hợp lệ
                continue
    
    # Xử lý phần còn lại trong buffer (nếu có)
    if buffer.strip():
        try:
            message = json.loads(buffer.strip())
            if isinstance(message, dict):
                yield message
        except json.JSONDecodeError:
            pass


def enhance_message_with_quality_instruction(message: str, quality: str) -> str:
    """
    Thêm instruction vào message để yêu cầu câu trả lời với chất lượng phù hợp.
    """
    quality_instructions = {
        "brief": "Hãy trả lời ngắn gọn, súc tích, tập trung vào thông tin chính.",
        "detailed": (
            "Hãy trả lời chi tiết, đầy đủ thông tin, giải thích rõ ràng các khái niệm, "
            "sự kiện, nhân vật. Bao gồm ngữ cảnh, nguyên nhân, diễn biến và ý nghĩa."
        ),
        "very_detailed": (
            "Hãy trả lời rất chi tiết và toàn diện. Bao gồm:\n"
            "- Giải thích đầy đủ các khái niệm, sự kiện, nhân vật\n"
            "- Ngữ cảnh lịch sử, nguyên nhân, diễn biến, kết quả và ý nghĩa\n"
            "- So sánh, liên hệ với các sự kiện khác nếu có\n"
            "- Ví dụ cụ thể và minh họa\n"
            "- Đảm bảo thông tin chính xác, đầy đủ và dễ hiểu"
        ),
    }
    
    instruction = quality_instructions.get(quality, quality_instructions["detailed"])
    return f"{message}\n\n[Yêu cầu: {instruction}]"


def stream_backend(
    message: str, max_retries: int = 1, chunk_size: int = 512
) -> Dict[str, Any]:
    """
    Gọi backend /chat ở chế độ stream để mô phỏng kiểu ChatGPT.
    Backend trả về NDJSON format với các message types: status, text, error, image.
    Trả về {"stream": Iterable[Dict[str, Any]]} hoặc {"error": "..."}.
    """
    backend_url: str = get_backend_url()
    is_healthy, health_msg = check_backend_health(backend_url, timeout=3)
    if not is_healthy:
        return {"error": health_msg, "backend_url": backend_url}

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                f"{backend_url}/chat",
                json={"message": message},
                timeout=90,
                stream=True,
                headers={
                    # Giúp giảm khả năng bị proxy/buffer, và nói rõ ta muốn stream text
                    "Accept": "application/x-ndjson",
                    "Cache-Control": "no-cache",
                },
            )
            resp.raise_for_status()

            def iter_raw_chunks() -> Iterable[str]:
                try:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            yield chunk.decode("utf-8", errors="ignore")
                finally:
                    resp.close()

            # Parse NDJSON stream thành các message objects
            parsed_stream = parse_ndjson_stream(iter_raw_chunks())
            return {"stream": parsed_stream, "backend_url": backend_url}

        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1)
                continue
            return {
                "error": (
                    "❌ **Lỗi kết nối:** Backend đã đóng kết nối đột ngột.\n\n"
                    "**Nguyên nhân có thể:**\n"
                    "- Backend đã crash hoặc tắt giữa chừng\n"
                    "- Backend đang quá tải và từ chối kết nối\n"
                    "- Firewall/antivirus chặn kết nối\n\n"
                    f"**Chi tiết:** `{str(e)}`\n\n"
                    f"**Backend URL:** `{backend_url}`"
                ),
            }
        except requests.exceptions.Timeout:
            return {
                "error": (
                    "⏱️ **Timeout:** Backend không phản hồi sau 90 giây.\n\n"
                    "Có thể backend đang xử lý request quá lâu hoặc đã crash.\n"
                    f"**Backend URL:** `{backend_url}`"
                ),
            }
        except requests.exceptions.HTTPError as e:
            return {
                "error": (
                    f"❌ **Lỗi HTTP {e.response.status_code}:** {str(e)}\n\n"
                    f"**Backend URL:** `{backend_url}`"
                ),
            }
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1)
                continue

    return {
        "error": (
            f"❌ **Lỗi sau {max_retries + 1} lần thử:** {str(last_error)}\n\n"
            f"**Backend URL:** `{backend_url}`\n\n"
            "Vui lòng kiểm tra backend có đang chạy không."
        ),
    }


def extract_answer_from_reply(reply: str) -> Tuple[str, Dict[str, Any]]:
    text: str = (reply or "").strip()
    meta: Dict[str, Any] = {}

    if not text:
        return "_(Không có nội dung trả về từ mô hình)_", meta

    if text[0] in "{[":
        try:
            data: Any = json.loads(text)
            if isinstance(data, dict):
                # Ưu tiên field "answer"
                answer_field = data.get("answer")
                if isinstance(answer_field, str) and answer_field.strip():
                    meta = {k: v for k, v in data.items() if k != "answer"}
                    return answer_field, meta

            # Nếu không có "answer" nhưng vẫn là JSON hợp lệ -> pretty JSON
            pretty_json: str = json.dumps(data, ensure_ascii=False, indent=2)
            return f"```json\n{pretty_json}\n```", meta
        except json.JSONDecodeError:
            # Không phải JSON hợp lệ -> xem như plain text
            pass

    return text, meta


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"]: List[ChatMessage] = [
            ChatMessage(
                role="assistant",
                content=(
                    "Xin chào 👋, mình là **Lịch Sử 10, 11, 12 AI Tutor**.\n\n"
                    "Bạn có thể:\n"
                    "- Hỏi giải thích các sự kiện, nhân vật lịch sử lớp 10, 11, 12.\n"
                    "- Yêu cầu tạo quiz / câu hỏi trắc nghiệm về một chủ đề lịch sử."
                ),
            )
        ]

    if "backend_url" not in st.session_state:
        st.session_state["backend_url"] = BACKEND_URL_DEFAULT
    
    # Trạng thái đang stream để disable chat input
    if "is_streaming" not in st.session_state:
        st.session_state["is_streaming"] = False
    
    # Flag để cancel stream
    if "cancel_stream" not in st.session_state:
        st.session_state["cancel_stream"] = False


def main() -> None:
    st.set_page_config(page_title="Lịch Sử 10, 11, 12 AI Tutor - Chatbot", page_icon="📚")
    init_session_state()

    st.title("📚 Lịch Sử 10, 11, 12 AI Tutor – Chatbot")

    # Hiển thị lịch sử hội thoại
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Hiển thị loading state và cancel button nếu đang stream
    if st.session_state.get("is_streaming", False):
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("⏹️ Dừng", key="cancel_stream_btn", use_container_width=True):
                st.session_state["cancel_stream"] = True
                st.session_state["is_streaming"] = False
                st.rerun()
        with col2:
            st.info("⏳ Đang xử lý câu trả lời...")
    
    # Ô nhập kiểu chatbot - chỉ hiển thị khi không đang stream
    if not st.session_state.get("is_streaming", False):
        prompt: str | None = st.chat_input(
            "Ask Anything You Want about History    "
        )
    else:
        prompt = None

    if prompt:
        user_text: str = prompt.strip()
        if not user_text:
            return

        # Lưu và hiển thị tin nhắn của user
        user_msg: ChatMessage = ChatMessage(role="user", content=user_text)
        st.session_state["messages"].append(user_msg)
        with st.chat_message("user"):
            st.markdown(user_text)

        # Đánh dấu đang stream
        st.session_state["is_streaming"] = True
        st.session_state["cancel_stream"] = False
        
        # Gọi backend và stream phản hồi giống ChatGPT
        with st.chat_message("assistant"):
            t0 = time.perf_counter()
            # Mặc định luôn dùng chế độ "detailed" để có câu trả lời tốt và chi tiết
            enhanced_message = enhance_message_with_quality_instruction(user_text, "detailed")
            stream_result = stream_backend(enhanced_message, chunk_size=512)
            reply_text: str = ""

            if "error" in stream_result:
                # Kết thúc stream khi có lỗi
                st.session_state["is_streaming"] = False
                reply_text = stream_result["error"]
                st.error(reply_text, icon="⚠️")
                with st.expander("💡 Gợi ý khắc phục", expanded=True):
                    st.markdown(
                        """
                        1. **Kiểm tra backend có đang chạy:**
                           ```bash
                           uvicorn app.main:app --reload --port 8000
                           ```
                        2. **Kiểm tra URL backend trong sidebar có đúng không**
                        3. **Kiểm tra log của backend** để xem có lỗi gì không
                        4. **Thử restart backend** nếu đang chạy
                        """
                    )
            else:
                stream = stream_result["stream"]
                placeholder = st.empty()
                status_placeholder = st.empty()
                image_status_placeholder = st.empty()  # Placeholder riêng cho image generation
                stats_placeholder = st.empty()

                # Chỉ collect text content để hiển thị
                text_content: List[str] = []
                status_messages: List[str] = []
                error_message: Optional[str] = None
                image_data: Optional[bytes] = None  # Lưu image để hiển thị cùng với text
                
                message_count: int = 0
                received_bytes: int = 0
                t_first: float | None = None

                for message in stream:
                    # Kiểm tra nếu user muốn cancel
                    if st.session_state.get("cancel_stream", False):
                        placeholder.markdown("".join(text_content) + "\n\n_(Đã dừng bởi người dùng)_")
                        st.session_state["is_streaming"] = False
                        break
                    
                    if not isinstance(message, dict):
                        continue
                    
                    msg_type: str = message.get("type", "")
                    message_count += 1
                    
                    if t_first is None:
                        t_first = time.perf_counter()
                    
                    # Xử lý các loại message khác nhau
                    if msg_type == "status":
                        status_msg = message.get("message", "")
                        if status_msg:
                            # Kiểm tra nếu đang generate image
                            status_lower = status_msg.lower()
                            if "image" in status_lower or "ảnh" in status_lower or "generating image" in status_lower:
                                # Hiển thị spinner message riêng cho image generation
                                image_status_placeholder.info(f"🖼️ {status_msg}")
                            else:
                                # Các status messages khác (searching, generating answer)
                                status_messages.append(status_msg)
                                status_placeholder.info(" | ".join(status_messages))
                    
                    elif msg_type == "text":
                        content = message.get("content", "")
                        if content:
                            text_content.append(content)
                            # Cập nhật text content theo thời gian thực
                            display_content = "".join(text_content)
                            placeholder.markdown(display_content)
                            # Khi bắt đầu có text, clear status messages cũ để không hiển thị nữa
                            if status_messages:
                                status_placeholder.empty()
                                status_messages = []  # Clear để không hiển thị lại
                    
                    elif msg_type == "error":
                        error_code = message.get("code", "UNKNOWN")
                        error_msg = message.get("message", "Unknown error")
                        error_message = f"❌ **Lỗi {error_code}:** {error_msg}"
                        status_placeholder.error(error_message)
                        image_status_placeholder.empty()  # Clear image status nếu có lỗi
                    
                    elif msg_type == "image":
                        base64_data = message.get("data", "")
                        if base64_data:
                            # Clear image status khi đã nhận được ảnh
                            image_status_placeholder.empty()
                            
                            # Decode và lưu image (sẽ hiển thị sau khi stream kết thúc)
                            try:
                                image_data = base64.b64decode(base64_data)
                                # Cập nhật text content nếu có để đảm bảo text được hiển thị trước
                                if text_content:
                                    placeholder.markdown("".join(text_content))
                            except Exception as e:
                                placeholder.warning(f"Không thể decode ảnh: {e}")
                                logger.error(f"Image decode error: {e}")
                                image_data = None

                    # Cập nhật thống kê
                    now = time.perf_counter()
                    ttfb_ms = (t_first - t0) * 1000 if t_first is not None else 0.0
                    elapsed_ms = (now - t0) * 1000
                    received_bytes += len(json.dumps(message, ensure_ascii=False).encode("utf-8"))
                    
                    # stats_placeholder.caption(
                    #     f"Streaming stats: messages={message_count} | bytes={received_bytes} | "
                    #     f"TTFB={ttfb_ms:.0f}ms | elapsed={elapsed_ms:.0f}ms"
                    # )

                # Kết thúc stream
                st.session_state["is_streaming"] = False
                
                # Clear tất cả status messages sau khi hoàn thành
                status_placeholder.empty()
                image_status_placeholder.empty()
                
                # Xử lý kết quả cuối cùng
                if error_message:
                    reply_text = error_message
                    placeholder.error(reply_text)
                elif text_content or image_data:
                    # Hiển thị text nếu có
                    if text_content:
                        reply_text = "".join(text_content).strip()
                        placeholder.markdown(reply_text)
                    else:
                        reply_text = ""
                    
                    # Hiển thị image nếu có (sau text, trong cùng chat message context)
                    if image_data:
                        try:
                            # Hiển thị image trong cùng chat message (width='stretch' để full width)
                            st.image(image_data, width='stretch')
                            if not reply_text:
                                reply_text = "_(Đã tạo ảnh)_"
                        except Exception as e:
                            st.warning(f"Không thể hiển thị ảnh: {e}")
                            logger.error(f"Image display error: {e}")
                    
                    if not reply_text:
                        reply_text = "_(Không có nội dung trả về)_"
                        placeholder.markdown(reply_text)
                else:
                    reply_text = "_(Không có nội dung trả về)_"
                    placeholder.markdown(reply_text)

        assistant_msg: ChatMessage = ChatMessage(role="assistant", content=reply_text)
        st.session_state["messages"].append(assistant_msg)


if __name__ == "__main__":
    main()
