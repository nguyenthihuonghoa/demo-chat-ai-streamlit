import os
import time
import json
import logging
from typing import Any, Dict, Iterable, List, Tuple, TypedDict, Optional

import requests
import streamlit as st
import base64

# Cấu hình logging
logger = logging.getLogger(__name__)

# --- 1. ĐỊNH NGHĨA KIỂU DỮ LIỆU ---
class ChatMessage(TypedDict):
    role: str  # "user" | "assistant"
    content: str
    image: Optional[bytes]  # Thêm trường này để lưu ảnh

# --- 2. CẤU HÌNH TRANG & CSS (FIX FOOTER) ---
st.set_page_config(page_title="Lịch Sử 10, 11, 12 AI Tutor", page_icon="📚")

# CSS để ẩn Footer, MainMenu và Header mặc định của Streamlit Cloud
hide_streamlit_style = """
<style>
    /* 1. Ẩn thanh Header trên cùng (cái vạch màu) */
    header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0%;
    }

    /* 2. Ẩn Footer mặc định "Made with Streamlit" */
    footer {
        visibility: hidden;
        height: 0%;
    }

    /* 3. Ẩn nút 3 chấm (Hamburger Menu) ở góc phải trên */
    div[data-testid="stToolbar"] {
        visibility: hidden;
        height: 0%;
    }

    /* 4. Ẩn các nút Decoration (góc phải trên) */
    div[data-testid="stDecoration"] {
        visibility: hidden;
        height: 0%;
    }

    /* 5. Ẩn nút "Manage app" / "Hosted with Streamlit" (Cái khó chịu nhất) */
    /* Cách này nhắm vào class chứa chữ 'viewerBadge' thường dùng cho nút góc phải dưới */
    div[class*="viewerBadge"] {
        display: none !important;
    }
    
    /* Ẩn luôn element cha của footer nếu cần */
    .stApp > footer {
        display: none !important;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Lấy URL mặc định từ biến môi trường (nếu có)
BACKEND_URL_DEFAULT: str = os.getenv("HISTORYBOT_BACKEND_URL", "http://localhost:8000")


# --- 3. CÁC HÀM XỬ LÝ BACKEND ---

def get_backend_url() -> str:
    # Ưu tiên lấy từ session state nếu người dùng đã nhập, nếu không thì dùng default
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


def parse_ndjson_stream(raw_chunks: Iterable[str]) -> Iterable[Dict[str, Any]]:
    """
    Parse NDJSON stream từ backend.
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
                continue
    
    # Xử lý phần còn lại trong buffer
    if buffer.strip():
        try:
            message = json.loads(buffer.strip())
            if isinstance(message, dict):
                yield message
        except json.JSONDecodeError:
            pass


def enhance_message_with_quality_instruction(message: str, quality: str) -> str:
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


def stream_backend(message: str, max_retries: int = 1, chunk_size: int = 512) -> Dict[str, Any]:
    backend_url: str = get_backend_url()
    
    # Check health nhanh trước khi gọi
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

            parsed_stream = parse_ndjson_stream(iter_raw_chunks())
            return {"stream": parsed_stream, "backend_url": backend_url}

        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1)
                continue
            return {"error": f"❌ Lỗi kết nối: Backend đóng kết nối đột ngột.\n{str(e)}"}
        except requests.exceptions.Timeout:
            return {"error": "⏱️ Timeout: Backend không phản hồi sau 90 giây."}
        except requests.exceptions.HTTPError as e:
            return {"error": f"❌ Lỗi HTTP {e.response.status_code}: {str(e)}"}
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1)
                continue

    return {"error": f"❌ Không thể kết nối sau {max_retries + 1} lần thử: {str(last_error)}"}


# --- 4. QUẢN LÝ SESSION STATE ---

def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Xin chào 👋, mình là **Lịch Sử 10, 11, 12 AI Tutor**.\n\n"
                    "Bạn có thể:\n"
                    "- Hỏi giải thích các sự kiện, nhân vật lịch sử lớp 10, 11, 12.\n"
                    "- Yêu cầu tạo quiz / câu hỏi trắc nghiệm về một chủ đề lịch sử."
                ),
                "image": None # Khởi tạo trường image là None
            }
        ]

    if "backend_url" not in st.session_state:
        st.session_state["backend_url"] = BACKEND_URL_DEFAULT
    
    if "is_streaming" not in st.session_state:
        st.session_state["is_streaming"] = False
    
    if "cancel_stream" not in st.session_state:
        st.session_state["cancel_stream"] = False


# --- 5. MAIN APP ---

def main() -> None:
    init_session_state()

    st.title("📚 Lịch Sử 10, 11, 12 AI Tutor – Chatbot")

    # --- HIỂN THỊ LỊCH SỬ ---
    # Phần này cực quan trọng: Hiển thị lại cả text và ảnh từ lịch sử
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if msg.get("content"):
                st.markdown(msg["content"])
            # Kiểm tra và hiển thị ảnh nếu có trong lịch sử
            if msg.get("image"):
                st.image(msg["image"], use_container_width=True)

    # --- NÚT DỪNG STREAM ---
    if st.session_state.get("is_streaming", False):
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("⏹️ Dừng", key="cancel_stream_btn", use_container_width=True):
                st.session_state["cancel_stream"] = True
                st.session_state["is_streaming"] = False
                st.rerun()
        with col2:
            st.info("⏳ Đang xử lý câu trả lời...")
    
    # --- INPUT USER ---
    if not st.session_state.get("is_streaming", False):
        prompt = st.chat_input("Ask Anything You Want about History")
    else:
        prompt = None

    if prompt:
        user_text = prompt.strip()
        if not user_text:
            return

        # 1. Hiển thị tin nhắn User
        user_msg: ChatMessage = {"role": "user", "content": user_text, "image": None}
        st.session_state["messages"].append(user_msg)
        with st.chat_message("user"):
            st.markdown(user_text)

        # 2. Bắt đầu xử lý Assistant response
        st.session_state["is_streaming"] = True
        st.session_state["cancel_stream"] = False
        
        with st.chat_message("assistant"):
            # Gọi API
            enhanced_message = enhance_message_with_quality_instruction(user_text, "detailed")
            stream_result = stream_backend(enhanced_message, chunk_size=512)
            
            reply_text: str = ""
            final_image_data: Optional[bytes] = None # Biến tạm để lưu ảnh nếu có

            if "error" in stream_result:
                st.session_state["is_streaming"] = False
                reply_text = stream_result["error"]
                st.error(reply_text, icon="⚠️")
            else:
                stream = stream_result["stream"]
                
                # Các placeholder để cập nhật UI realtime
                placeholder = st.empty()
                status_placeholder = st.empty()
                image_status_placeholder = st.empty()
                
                text_content: List[str] = []
                status_messages: List[str] = []
                
                try:
                    for message in stream:
                        # Check cancel
                        if st.session_state.get("cancel_stream", False):
                            placeholder.markdown("".join(text_content) + "\n\n_(Đã dừng bởi người dùng)_")
                            reply_text = "".join(text_content) + "\n\n_(Đã dừng bởi người dùng)_"
                            st.session_state["is_streaming"] = False
                            break
                        
                        msg_type = message.get("type", "")
                        
                        if msg_type == "status":
                            status_msg = message.get("message", "")
                            if status_msg:
                                s_lower = status_msg.lower()
                                if any(x in s_lower for x in ["image", "ảnh", "generating image"]):
                                    image_status_placeholder.info(f"🖼️ {status_msg}")
                                else:
                                    status_messages.append(status_msg)
                                    status_placeholder.info(" | ".join(status_messages))
                        
                        elif msg_type == "text":
                            content = message.get("content", "")
                            if content:
                                text_content.append(content)
                                placeholder.markdown("".join(text_content))
                                # Có text rồi thì clear status search đi cho gọn
                                if status_messages:
                                    status_placeholder.empty()
                                    status_messages = []

                        elif msg_type == "error":
                            err_msg = message.get("message", "Unknown error")
                            st.error(f"Lỗi: {err_msg}")
                        
                        elif msg_type == "image":
                            base64_data = message.get("data", "")
                            if base64_data:
                                image_status_placeholder.empty() # Xóa trạng thái "đang tạo ảnh"
                                try:
                                    # Decode và lưu vào biến tạm
                                    final_image_data = base64.b64decode(base64_data)
                                    
                                    # Render lại Text + Image ngay lập tức
                                    if text_content:
                                        placeholder.markdown("".join(text_content))
                                    
                                    # Dùng use_container_width=True thay cho width='stretch'
                                    st.image(final_image_data, use_container_width=True)
                                    
                                except Exception as e:
                                    st.warning(f"Không thể hiển thị ảnh: {e}")

                    # Kết thúc vòng lặp stream
                    reply_text = "".join(text_content).strip()
                    if not reply_text and not final_image_data:
                        reply_text = "_(Không có nội dung trả về)_"
                        placeholder.markdown(reply_text)

                except Exception as e:
                    st.error(f"Có lỗi xảy ra trong quá trình stream: {e}")
                    reply_text += f"\n\n(Lỗi: {e})"
                
                finally:
                    # Dọn dẹp các placeholder trạng thái
                    status_placeholder.empty()
                    image_status_placeholder.empty()

            # --- LƯU VÀO SESSION STATE (QUAN TRỌNG) ---
            st.session_state["is_streaming"] = False
            
            assistant_msg: ChatMessage = {
                "role": "assistant", 
                "content": reply_text,
                "image": final_image_data # Lưu bytes ảnh vào lịch sử
            }
            st.session_state["messages"].append(assistant_msg)

if __name__ == "__main__":
    main()
