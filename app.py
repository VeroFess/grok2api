from __future__ import annotations

import os
import json
import uuid
import time
import base64
import sys
import inspect
import secrets
import asyncio
import contextvars
import functools
import re
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Union, Tuple, Iterator
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import requests
from flask import (
    Flask,
    request,
    Response,
    jsonify,
    stream_with_context,
    render_template,
    redirect,
    session,
)
from curl_cffi import requests as curl_requests
from werkzeug.middleware.proxy_fix import ProxyFix


class ModelType(Enum):
    """Supported Grok model types with new architecture."""

    GROK_3 = "grok-3"
    GROK_4 = "grok-4"

    GROK_AUTO = "grok-auto"  # grok-3 + MODEL_MODE_AUTO
    GROK_FAST = "grok-fast"  # grok-3 + MODEL_MODE_FAST
    GROK_EXPERT = "grok-expert"  # grok-4 + MODEL_MODE_EXPERT
    GROK_SEARCH = "grok-deepsearch"  # grok-4 + MODEL_MODE_EXPERT + workspaceIds
    GROK_IMAGE = "grok-image"  # grok-4 + MODEL_MODE_EXPERT + enableImageGeneration


class TokenType(Enum):
    """Token privilege levels."""

    NORMAL = "normal"
    SUPER = "super"


class ResponseState(Enum):
    """Response processing states."""

    IDLE = "idle"
    THINKING = "thinking"
    GENERATING_IMAGE = "generating_image"
    COMPLETE = "complete"


MESSAGE_LENGTH_LIMIT = 40000
MAX_FILE_ATTACHMENTS = 4
DEFAULT_REQUEST_TIMEOUT = 120000
MAX_RETRY_ATTEMPTS = 3
BASE_RETRY_DELAY = 1.0


BASE_HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Content-Type": "text/plain;charset=UTF-8",
    "Connection": "keep-alive",
    "Origin": "https://grok.com",
    "Priority": "u=1, i",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Sec-Ch-Ua": '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Baggage": "sentry-public_key=b311e0f2690c81f25e2c4cf6d4f7ce1c",
}


def get_dynamic_headers(
    method: str = "POST",
    pathname: str = "/rest/app-chat/conversations/new",
    config: Optional["ConfigurationManager"] = None,
) -> Dict[str, str]:
    headers = BASE_HEADERS.copy()

    headers["x-xai-request-id"] = str(uuid.uuid4())
    headers["x-statsig-id"] = (
        "ZTpUeXBlRXJyb3I6IENhbm5vdCByZWFkIHByb3BlcnRpZXMgb2YgdW5kZWZpbmVkIChyZWFkaW5nICdjaGlsZE5vZGVzJyk="
    )
    return headers


class GrokApiException(Exception):
    """Base exception for Grok API errors."""

    def __init__(self, message: str, error_code: str = "UNKNOWN_ERROR"):
        super().__init__(message)
        self.error_code = error_code


class TokenException(GrokApiException):
    """Token-related exceptions."""

    pass


class ValidationException(GrokApiException):
    """Input validation exceptions."""

    pass


class RateLimitException(GrokApiException):
    """Rate limiting exceptions."""

    pass


@dataclass
class TokenCredential:
    """Represents a token credential with validation."""

    sso_token: str
    token_type: TokenType

    def __post_init__(self):
        """Validate token format."""
        if not self.sso_token or not self.sso_token.strip():
            raise ValidationException("SSO token cannot be empty")
        if "sso=" not in self.sso_token:
            raise ValidationException("Invalid SSO token format")
        try:
            parts = self.sso_token.split("sso=")
            if len(parts) < 2 or not parts[1]:
                raise ValidationException(
                    "Invalid SSO token format: missing value after 'sso='"
                )
        except Exception as e:
            raise ValidationException(f"Invalid SSO token format: {e}")

    @classmethod
    def from_raw_token(
        cls, raw_token: str, token_type: TokenType = TokenType.NORMAL
    ) -> "TokenCredential":
        """Create TokenCredential from raw SSO value."""
        if not raw_token or not raw_token.strip():
            raise ValidationException("Raw token cannot be empty")

        sanitized_token = raw_token.strip()
        if ";" in sanitized_token:
            raise ValidationException("Raw token contains invalid character (';')")

        formatted_token = f"sso-rw={sanitized_token};sso={sanitized_token}"
        return cls(formatted_token, token_type)

    def extract_sso_value(self) -> str:
        """Extract the SSO value from the token."""
        try:
            return self.sso_token.split("sso=")[1].split(";")[0]
        except (IndexError, AttributeError) as e:
            raise TokenException(f"Failed to parse SSO token: {self.sso_token}") from e


@dataclass
class GeneratedImage:
    """Represents a generated image with metadata."""

    url: str
    base_url: str = "https://assets.grok.com"
    cookies: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Validate image data."""
        if not self.url:
            raise ValidationException("Image URL cannot be empty")


@dataclass
class ProcessingState:
    """Immutable state for response processing."""

    is_thinking: bool = False
    is_generating_image: bool = False
    image_generation_phase: int = 0

    def with_thinking(self, thinking: bool) -> "ProcessingState":
        """Return new state with updated thinking status."""
        return ProcessingState(
            thinking, self.is_generating_image, self.image_generation_phase
        )

    def with_image_generation(
        self, generating: bool, phase: int = 0
    ) -> "ProcessingState":
        """Return new state with updated image generation status."""
        return ProcessingState(self.is_thinking, generating, phase)


@dataclass
class ModelResponse:
    """Enhanced model response with proper validation and transformation."""

    response_id: str
    message: str
    sender: str
    create_time: str
    parent_response_id: str
    manual: bool
    partial: bool
    shared: bool
    query: str
    query_type: str
    web_search_results: List[Any] = field(default_factory=list)
    xpost_ids: List[Any] = field(default_factory=list)
    xposts: List[Any] = field(default_factory=list)
    generated_images: List[GeneratedImage] = field(default_factory=list)
    image_attachments: List[Any] = field(default_factory=list)
    file_attachments: List[Any] = field(default_factory=list)
    card_attachments_json: List[Any] = field(default_factory=list)
    file_uris: List[Any] = field(default_factory=list)
    file_attachments_metadata: List[Any] = field(default_factory=list)
    is_control: bool = False
    steps: List[Any] = field(default_factory=list)
    media_types: List[Any] = field(default_factory=list)

    @classmethod
    def from_api_response(
        cls, data: Dict[str, Any], enable_artifact_files: bool = False
    ) -> "ModelResponse":
        """Create ModelResponse from API response data with validation."""
        try:
            response_id = str(data.get("responseId", ""))
            sender = str(data.get("sender", ""))
            create_time = str(data.get("createTime", ""))
            parent_response_id = str(data.get("parentResponseId", ""))
            query = str(data.get("query", ""))
            query_type = str(data.get("queryType", ""))

            manual = bool(data.get("manual", False))
            partial = bool(data.get("partial", False))
            shared = bool(data.get("shared", False))
            is_control = bool(data.get("isControl", False))

            raw_message = data.get("message", "")
            processed_message = cls._transform_xai_artifacts(str(raw_message))

            generated_images = []
            for image_url in data.get("generatedImageUrls", []):
                if image_url:
                    generated_images.append(GeneratedImage(url=str(image_url)))

            return cls(
                response_id=response_id,
                message=processed_message,
                sender=sender,
                create_time=create_time,
                parent_response_id=parent_response_id,
                manual=manual,
                partial=partial,
                shared=shared,
                query=query,
                query_type=query_type,
                web_search_results=data.get("webSearchResults", []),
                xpost_ids=data.get("xpostIds", []),
                xposts=data.get("xposts", []),
                generated_images=generated_images,
                image_attachments=data.get("imageAttachments", []),
                file_attachments=data.get("fileAttachments", []),
                card_attachments_json=data.get("cardAttachmentsJson", []),
                file_uris=data.get("fileUris", []),
                file_attachments_metadata=data.get("fileAttachmentsMetadata", []),
                is_control=is_control,
                steps=data.get("steps", []),
                media_types=data.get("mediaTypes", []),
            )
        except Exception as e:
            print(f"Failed to create ModelResponse: {e}")
            return cls(
                response_id="",
                message="Error processing response",
                sender="system",
                create_time=str(int(time.time())),
                parent_response_id="",
                manual=False,
                partial=False,
                shared=False,
                query="",
                query_type="",
            )

    @staticmethod
    def _transform_xai_artifacts(text: str) -> str:
        """
        Transform xaiArtifact blocks to proper markdown code blocks.
        Comprehensive version that handles all xaiArtifact formats including:
        1. <xaiArtifact contentType="text/..."> blocks → ```<lang>\ncode\n```
        2. ```x-<lang>src format → ```<lang>
        3. ```x-<lang> format → ```<lang>
        4. Any xaiArtifact with artifact_id, title, etc.
        5. Self-closing xaiArtifact tags
        """
        if not text:
            return text

        def replace_artifact_with_content(match):
            full_match = match.group(0)
            content = match.group(1).strip() if match.group(1) else ""

            content_type_match = re.search(r'contentType="([^"]+)"', full_match)
            if content_type_match:
                content_type = content_type_match.group(1).strip()
                if "/" in content_type:
                    lang = content_type.split("/")[-1]
                else:
                    lang = content_type

                if content:
                    return f"```{lang}\n{content}\n```"
                else:
                    return ""
            else:
                return content

        text = re.sub(
            r"<xaiArtifact[^>]*?>(.*?)</xaiArtifact>",
            replace_artifact_with_content,
            text,
            flags=re.DOTALL,
        )

        text = re.sub(r"<xaiArtifact[^>]*?/>", "", text)

        text = re.sub(r"<xaiArtifact[^>]*>", "", text)
        text = re.sub(r"</xaiArtifact>", "", text)

        text = re.sub(
            r"```x-([a-zA-Z0-9_+-]+)src\b", lambda m: f"```{m.group(1)}", text
        )

        text = re.sub(
            r"```x-([a-zA-Z0-9_+-]+)\b(?![a-zA-Z0-9_-]*src)",
            lambda m: f"```{m.group(1)}",
            text,
        )

        return text


@dataclass
class GrokResponse:
    """Complete Grok API response wrapper."""

    model_response: ModelResponse
    is_thinking: bool = False
    is_soft_stop: bool = False
    response_id: str = ""
    conversation_id: Optional[str] = None
    title: Optional[str] = None
    conversation_create_time: Optional[str] = None
    conversation_modify_time: Optional[str] = None
    temporary: Optional[bool] = None
    error: Optional[str] = None
    error_code: Optional[Union[int, str]] = None

    @classmethod
    def from_api_response(
        cls, data: Dict[str, Any], enable_artifact_files: bool = False
    ) -> "GrokResponse":
        """Create GrokResponse from API response data."""
        try:
            error = data.get("error")
            error_code = data.get("error_code")
            result = data.get("result", {})
            response_data = result.get("response", {})

            model_response = ModelResponse.from_api_response(
                response_data.get("modelResponse", {}), enable_artifact_files
            )

            is_thinking = bool(response_data.get("isThinking", False))
            is_soft_stop = bool(response_data.get("isSoftStop", False))
            response_id = str(response_data.get("responseId", ""))

            conversation_id = response_data.get("conversationId")
            new_title = result.get("newTitle") or result.get("title")
            title = new_title if new_title else None
            conversation_create_time = response_data.get("createTime")
            conversation_modify_time = response_data.get("modifyTime")
            temporary = response_data.get("temporary")

            return cls(
                model_response=model_response,
                is_thinking=is_thinking,
                is_soft_stop=is_soft_stop,
                response_id=response_id,
                conversation_id=conversation_id,
                title=title,
                conversation_create_time=conversation_create_time,
                conversation_modify_time=conversation_modify_time,
                temporary=temporary,
                error=error,
                error_code=error_code,
            )
        except Exception as e:
            error_msg = str(e)
            return cls(
                model_response=ModelResponse.from_api_response({}),
                error=error_msg,
                error_code="RESPONSE_PARSING_ERROR",
            )


class ConfigurationManager:
    """Centralized configuration management with validation."""

    def __init__(self):
        """Initialize configuration with environment variables."""
        self.data_dir = Path("/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._config = self._load_configuration()
        self._validate_configuration()

    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""

        model_mapping = {}

        for model in ModelType:
            alias = model.value
            if alias in ["grok-3", "grok-auto"]:
                model_mapping[alias] = "grok-3"
            elif alias in ["grok-fast"]:
                model_mapping[alias] = "grok-3"
            elif alias in ["grok-4", "grok-expert"]:
                model_mapping[alias] = "grok-4"
            elif alias in ["grok-search", "grok-deepsearch"]:
                model_mapping[alias] = "grok-4"
            elif alias in ["grok-image", "grok-draw"]:
                model_mapping[alias] = "grok-4"
            else:

                model_mapping[alias] = alias

        return {
            "MODELS": model_mapping,
            "API": {
                "IS_TEMP_CONVERSATION": self._get_bool_env(
                    "IS_TEMP_CONVERSATION", True
                ),
                "IS_CUSTOM_SSO": self._get_bool_env("IS_CUSTOM_SSO", False),
                "BASE_URL": "https://grok.com",
                "API_KEY": os.environ.get("API_KEY", "sk-123456"),
                "RETRY_TIME": 1000,
                "PROXY": os.environ.get("PROXY"),
                "DISABLE_DYNAMIC_HEADERS": self._get_bool_env(
                    "DISABLE_DYNAMIC_HEADERS", False
                ),
            },
            "ADMIN": {
                "MANAGER_SWITCH": os.environ.get("MANAGER_SWITCH"),
                "PASSWORD": os.environ.get("ADMINPASSWORD"),
            },
            "SERVER": {
                "CF_CLEARANCE": os.environ.get("CF_CLEARANCE"),
                "PORT": int(os.environ.get("PORT", 5200)),
            },
            "RETRY": {
                "RETRYSWITCH": False,
                "MAX_ATTEMPTS": MAX_RETRY_ATTEMPTS,
            },
            "TOKEN_STATUS_FILE": str(self.data_dir / "token_status.json"),
            "SHOW_THINKING": self._get_bool_env("SHOW_THINKING", False),
            "SHOW_SEARCH_RESULTS": self._get_bool_env("SHOW_SEARCH_RESULTS", True),
            "IS_SUPER_GROK": self._get_bool_env("IS_SUPER_GROK", False),
            "FILTERED_TAGS": self._get_list_env(
                "FILTERED_TAGS",
                [
                    "xaiartifact",
                    "xai:tool_usage_card",
                    "grok:render",
                    "details",
                    "summary",
                ],
            ),
            "TAG_CONFIG": self._get_tag_config(),
            "CONTENT_TYPE_MAPPINGS": self._get_content_type_mappings(),
        }

    def _get_bool_env(self, key: str, default: bool = False) -> bool:
        """Get boolean environment variable."""
        return os.environ.get(key, str(default)).lower() == "true"

    def _get_list_env(self, key: str, default: List[str]) -> List[str]:
        """Get comma-separated list from environment variable."""
        value = os.environ.get(key)
        if not value:
            return default
        return [tag.strip() for tag in value.split(",") if tag.strip()]

    def _get_content_type_mappings(self) -> Dict[str, Dict[str, str]]:
        """Get content type mappings from environment or defaults."""
        mappings_env = os.environ.get("CONTENT_TYPE_MAPPINGS")
        if mappings_env:
            try:
                return json.loads(mappings_env)
            except json.JSONDecodeError:
                print("Invalid CONTENT_TYPE_MAPPINGS JSON, using defaults")

        return {
            "text/plain": {"stag": "```", "etag": "```"},
            "text/markdown": {"stag": "", "etag": ""},
            "application/json": {"stag": "```json\n", "etag": "\n```"},
            "text/javascript": {"stag": "```javascript\n", "etag": "\n```"},
            "text/python": {"stag": "```python\n", "etag": "\n```"},
            "text/html": {"stag": "```html\n", "etag": "\n```"},
            "text/css": {"stag": "```css\n", "etag": "\n```"},
            "text/xml": {"stag": "```xml\n", "etag": "\n```"},
            "application/xml": {"stag": "```xml\n", "etag": "\n```"},
            "text/yaml": {"stag": "```yaml\n", "etag": "\n```"},
            "application/yaml": {"stag": "```yaml\n", "etag": "\n```"},
            "text/x-yaml": {"stag": "```yaml\n", "etag": "\n```"},
            "text/sql": {"stag": "```sql\n", "etag": "\n```"},
            "application/sql": {"stag": "```sql\n", "etag": "\n```"},
            "text/x-sql": {"stag": "```sql\n", "etag": "\n```"},
            "text/typescript": {"stag": "```typescript\n", "etag": "\n```"},
            "application/typescript": {"stag": "```typescript\n", "etag": "\n```"},
            "text/x-typescript": {"stag": "```typescript\n", "etag": "\n```"},
            "text/jsx": {"stag": "```jsx\n", "etag": "\n```"},
            "text/x-jsx": {"stag": "```jsx\n", "etag": "\n```"},
            "text/tsx": {"stag": "```tsx\n", "etag": "\n```"},
            "text/x-tsx": {"stag": "```tsx\n", "etag": "\n```"},
            "text/java": {"stag": "```java\n", "etag": "\n```"},
            "application/java": {"stag": "```java\n", "etag": "\n```"},
            "text/x-java": {"stag": "```java\n", "etag": "\n```"},
            "text/csharp": {"stag": "```csharp\n", "etag": "\n```"},
            "text/x-csharp": {"stag": "```csharp\n", "etag": "\n```"},
            "application/x-csharp": {"stag": "```csharp\n", "etag": "\n```"},
            "text/cpp": {"stag": "```cpp\n", "etag": "\n```"},
            "text/x-c++": {"stag": "```cpp\n", "etag": "\n```"},
            "application/x-cpp": {"stag": "```cpp\n", "etag": "\n```"},
            "text/c": {"stag": "```c\n", "etag": "\n```"},
            "text/x-c": {"stag": "```c\n", "etag": "\n```"},
            "text/go": {"stag": "```go\n", "etag": "\n```"},
            "text/x-go": {"stag": "```go\n", "etag": "\n```"},
            "application/x-go": {"stag": "```go\n", "etag": "\n```"},
            "text/rust": {"stag": "```rust\n", "etag": "\n```"},
            "text/x-rust": {"stag": "```rust\n", "etag": "\n```"},
            "application/x-rust": {"stag": "```rust\n", "etag": "\n```"},
            "text/php": {"stag": "```php\n", "etag": "\n```"},
            "application/x-php": {"stag": "```php\n", "etag": "\n```"},
            "text/ruby": {"stag": "```ruby\n", "etag": "\n```"},
            "application/x-ruby": {"stag": "```ruby\n", "etag": "\n```"},
            "text/swift": {"stag": "```swift\n", "etag": "\n```"},
            "text/x-swift": {"stag": "```swift\n", "etag": "\n```"},
            "text/kotlin": {"stag": "```kotlin\n", "etag": "\n```"},
            "text/x-kotlin": {"stag": "```kotlin\n", "etag": "\n```"},
            "text/scala": {"stag": "```scala\n", "etag": "\n```"},
            "text/x-scala": {"stag": "```scala\n", "etag": "\n```"},
            "text/bash": {"stag": "```bash\n", "etag": "\n```"},
            "text/x-bash": {"stag": "```bash\n", "etag": "\n```"},
            "application/x-bash": {"stag": "```bash\n", "etag": "\n```"},
            "text/shell": {"stag": "```bash\n", "etag": "\n```"},
            "text/x-shell": {"stag": "```bash\n", "etag": "\n```"},
            "application/x-shell": {"stag": "```bash\n", "etag": "\n```"},
            "text/powershell": {"stag": "```powershell\n", "etag": "\n```"},
            "text/x-powershell": {"stag": "```powershell\n", "etag": "\n```"},
            "application/x-powershell": {"stag": "```powershell\n", "etag": "\n```"},
            "text/dockerfile": {"stag": "```dockerfile\n", "etag": "\n```"},
            "text/x-dockerfile": {"stag": "```dockerfile\n", "etag": "\n```"},
            "application/x-dockerfile": {"stag": "```dockerfile\n", "etag": "\n```"},
            "text/toml": {"stag": "```toml\n", "etag": "\n```"},
            "application/toml": {"stag": "```toml\n", "etag": "\n```"},
            "text/ini": {"stag": "```ini\n", "etag": "\n```"},
            "text/x-ini": {"stag": "```ini\n", "etag": "\n```"},
            "application/x-ini": {"stag": "```ini\n", "etag": "\n```"},
            "text/properties": {"stag": "```properties\n", "etag": "\n```"},
            "text/x-properties": {"stag": "```properties\n", "etag": "\n```"},
            "text/csv": {"stag": "```csv\n", "etag": "\n```"},
            "application/csv": {"stag": "```csv\n", "etag": "\n```"},
            "text/x-csv": {"stag": "```csv\n", "etag": "\n```"},
            "text/log": {"stag": "```\n", "etag": "\n```"},
            "application/x-log": {"stag": "```\n", "etag": "\n```"},
            "text/x-log": {"stag": "```\n", "etag": "\n```"},
            "application/x-httpd-php": {"stag": "```php\n", "etag": "\n```"},
            "text/x-python": {"stag": "```python\n", "etag": "\n```"},
            "application/x-python": {"stag": "```python\n", "etag": "\n```"},
            "text/x-javascript": {"stag": "```javascript\n", "etag": "\n```"},
            "application/javascript": {"stag": "```javascript\n", "etag": "\n```"},
            "text/ecmascript": {"stag": "```javascript\n", "etag": "\n```"},
            "application/ecmascript": {"stag": "```javascript\n", "etag": "\n```"},
            "text/jscript": {"stag": "```javascript\n", "etag": "\n```"},
            "application/x-javascript": {"stag": "```javascript\n", "etag": "\n```"},
            "text/vbscript": {"stag": "```vbscript\n", "etag": "\n```"},
            "application/x-vbscript": {"stag": "```vbscript\n", "etag": "\n```"},
            "text/x-markdown": {"stag": "", "etag": ""},
            "application/x-markdown": {"stag": "", "etag": ""},
            "text/x-web-markdown": {"stag": "", "etag": ""},
            "code/python": {"stag": "```python\n", "etag": "\n```"},
            "code/javascript": {"stag": "```javascript\n", "etag": "\n```"},
            "code/typescript": {"stag": "```typescript\n", "etag": "\n```"},
            "code/html": {"stag": "```html\n", "etag": "\n```"},
            "code/css": {"stag": "```css\n", "etag": "\n```"},
            "code/json": {"stag": "```json\n", "etag": "\n```"},
            "code/xml": {"stag": "```xml\n", "etag": "\n```"},
            "code/yaml": {"stag": "```yaml\n", "etag": "\n```"},
            "code/sql": {"stag": "```sql\n", "etag": "\n```"},
            "code/bash": {"stag": "```bash\n", "etag": "\n```"},
            "code/shell": {"stag": "```bash\n", "etag": "\n```"},
            "code/dockerfile": {"stag": "```dockerfile\n", "etag": "\n```"},
            "text/code": {"stag": "```\n", "etag": "\n```"},
            "application/code": {"stag": "```\n", "etag": "\n```"},
            "text/source": {"stag": "```\n", "etag": "\n```"},
            "application/source": {"stag": "```\n", "etag": "\n```"},
        }

    def _get_tag_config(self) -> Dict[str, Dict[str, Any]]:
        """Get tag configuration from environment or defaults."""
        tag_config_env = os.environ.get("TAG_CONFIG")
        if tag_config_env:
            try:
                return json.loads(tag_config_env)
            except json.JSONDecodeError:
                print("Invalid TAG_CONFIG JSON, using defaults")

        filtered_tags = self._get_list_env(
            "FILTERED_TAGS",
            ["xaiartifact", "xai:tool_usage_card", "grok:render", "details", "summary"],
        )
        default_config = {}

        for tag in filtered_tags:
            if tag.lower() in ["xai:tool_usage_card", "grok:render"]:
                default_config[tag.lower()] = {"behavior": "remove_all"}
            else:
                default_config[tag.lower()] = {"behavior": "preserve_content"}

        if not default_config:
            default_config = {
                "xaiartifact": {"behavior": "preserve_content"},
                "xai:tool_usage_card": {"behavior": "remove_all"},
                "grok:render": {"behavior": "remove_all"},
                "details": {"behavior": "preserve_content"},
                "summary": {"behavior": "preserve_content"},
            }

        return default_config

    def _validate_configuration(self) -> None:
        """Validate configuration settings."""
        issues = []

        if not os.environ.get("API_KEY"):
            issues.append("Missing required environment variable: API_KEY")

        if not self._config["API"]["IS_CUSTOM_SSO"]:
            sso_env = os.environ.get("SSO", "")
            sso_super_env = os.environ.get("SSO_SUPER", "")
            if not sso_env and not sso_super_env:
                issues.append(
                    "No SSO tokens configured. Set SSO or SSO_SUPER environment variables."
                )

        proxy = self._config["API"]["PROXY"]
        if proxy and not any(
            proxy.startswith(p) for p in ["http://", "https://", "socks5://"]
        ):
            issues.append(f"Invalid proxy format: {proxy}")

        if issues:
            for issue in issues:
                print(f"Configuration issue: {issue}")
        else:
            print("Configuration validation passed")

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key_path.split(".")
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    @property
    def models(self) -> Dict[str, str]:
        """Get supported models mapping."""
        return self._config["MODELS"]

    @property
    def data_directory(self) -> Path:
        """Get data directory path."""
        return self.data_dir


class UtilityFunctions:
    """Collection of utility functions for common operations."""

    @staticmethod
    def get_proxy_configuration(proxy_url: Optional[str]) -> Dict[str, Any]:
        """Get proxy configuration for requests."""
        if not proxy_url:
            return {}

        print(f"Using proxy: {proxy_url}")

        if proxy_url.startswith("socks5://"):
            proxy_config: Dict[str, Any] = {"proxy": proxy_url}

            if "@" in proxy_url:
                auth_part = proxy_url.split("@")[0].split("://")[1]
                if ":" in auth_part:
                    username, password = auth_part.split(":", 1)
                    proxy_config["proxy_auth"] = (username, password)

            return proxy_config
        else:
            return {"proxies": {"https": proxy_url, "http": proxy_url}}

    @staticmethod
    def organize_search_results(search_results: Dict[str, Any]) -> str:
        """Format search results for display."""
        if not search_results or "results" not in search_results:
            return ""

        results = search_results["results"]
        formatted_results = []

        for index, result in enumerate(results):
            title = result.get("title", "未知标题")
            url = result.get("url", "#")
            preview = result.get("preview", "无预览内容")

            formatted_result = (
                f"\r\n<details><summary>资料[{index}]: {title}</summary>\r\n"
                f"{preview}\r\n\n[Link]({url})\r\n</details>"
            )
            formatted_results.append(formatted_result)

        return "\n\n".join(formatted_results)

    @staticmethod
    def parse_error_response(response_text: str) -> Dict[str, Any]:
        """Parse error response with structured handling."""
        if not response_text or not response_text.strip():
            return {
                "error_code": "EMPTY_RESPONSE",
                "error": "Empty or invalid response received",
                "details": [],
            }

        try:
            try:
                response = json.loads(response_text)
                if isinstance(response, dict):
                    if "error" in response:
                        error = response["error"]
                        if isinstance(error, dict):
                            return {
                                "error_code": error.get("code"),
                                "error": error.get("message") or response_text,
                                "details": (
                                    error.get("details", [])
                                    if isinstance(error.get("details"), list)
                                    else []
                                ),
                            }
                        else:
                            return {
                                "error_code": "Unknown",
                                "error": str(error),
                                "details": [],
                            }
                    elif "message" in response:
                        return {
                            "error_code": response.get("code"),
                            "error": response.get("message") or response_text,
                            "details": (
                                response.get("details", [])
                                if isinstance(response.get("details"), list)
                                else []
                            ),
                        }
                    else:
                        return {
                            "error_code": "Unknown",
                            "error": response_text,
                            "details": [],
                        }
            except json.JSONDecodeError:
                pass

            if " - " in response_text:
                json_str = response_text.split(" - ", 1)[1]
                response = json.loads(json_str)

                if isinstance(response, dict):
                    if "error" in response:
                        error = response["error"]
                        if isinstance(error, dict):
                            return {
                                "error_code": error.get("code"),
                                "error": error.get("message") or response_text,
                                "details": (
                                    error.get("details", [])
                                    if isinstance(error.get("details"), list)
                                    else []
                                ),
                            }
                        else:
                            return {
                                "error_code": "Unknown",
                                "error": str(error),
                                "details": [],
                            }
                    elif "message" in response:
                        return {
                            "error_code": response.get("code"),
                            "error": response.get("message") or response_text,
                            "details": (
                                response.get("details", [])
                                if isinstance(response.get("details"), list)
                                else []
                            ),
                        }

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            print(f"Error parsing error response: {e}")

        return {
            "error_code": "Unknown",
            "error": response_text or "Unknown error occurred",
            "details": [],
        }

    @staticmethod
    def create_retry_decorator(
        max_attempts: int = MAX_RETRY_ATTEMPTS, base_delay: float = BASE_RETRY_DELAY
    ):
        """Create retry decorator with exponential backoff."""

        def retry_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                attempts = 0
                last_error = None

                while attempts < max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        attempts += 1
                        last_error = e

                        if attempts >= max_attempts:
                            print(
                                f"All retries failed ({max_attempts} attempts): {e}",
                                "RetryMechanism",
                            )
                            raise e

                        delay = min(base_delay * (2 ** (attempts - 1)), 60)
                        print(
                            f"Retry {attempts}/{max_attempts}, delay {delay}s: {e}",
                            "RetryMechanism",
                        )
                        time.sleep(delay)

                raise last_error or Exception("Retry mechanism failed unexpectedly")

            return wrapper

        return retry_decorator

    @staticmethod
    async def run_in_thread_pool(func, *args, **kwargs):
        """Run synchronous function in thread pool for async compatibility."""
        try:
            loop = asyncio.get_running_loop()
            ctx = contextvars.copy_context()
            func_call = functools.partial(ctx.run, func, *args, **kwargs)
            return await loop.run_in_executor(None, func_call)
        except RuntimeError:
            with ThreadPoolExecutor() as executor:
                future = executor.submit(func, *args, **kwargs)
                return future.result()

    @staticmethod
    def create_structured_error_response(
        error_data: Union[str, Dict[str, Any]], status_code: int = 500
    ) -> Tuple[Dict[str, Any], int]:
        """Create structured error response."""
        if isinstance(error_data, str):
            error_data = UtilityFunctions.parse_error_response(error_data)

        error_message = error_data.get("error", "Unknown error")
        error_code = error_data.get("error_code")
        error_details = error_data.get("details", [])

        if not error_message or error_message.strip() == "":
            error_message = "An error occurred while processing the request"

        return {
            "error": {
                "message": str(error_message),
                "type": "server_error",
                "code": str(error_code),
                "details": list(error_details) if error_details else [],
            }
        }, status_code


@dataclass
class TokenEntry:
    """Represents a single token entry with usage tracking."""

    credential: TokenCredential
    max_request_count: int
    request_count: int
    added_time: int
    start_call_time: Optional[int] = None

    def is_available(self) -> bool:
        """Check if token is available for use. Allow over-limit usage with warnings."""
        return self.request_count < (self.max_request_count * 100)

    def can_be_reset(self, expiration_time_ms: int, current_time_ms: int) -> bool:
        """Check if token can be reset based on expiration time."""
        if not self.start_call_time:
            return False
        return current_time_ms - self.start_call_time >= expiration_time_ms

    def use_token(self) -> None:
        """Mark token as used."""
        if not self.start_call_time:
            self.start_call_time = int(time.time() * 1000)
        self.request_count += 1

    def reset_usage(self) -> None:
        """Reset token usage counters."""
        self.request_count = 0
        self.start_call_time = None


@dataclass
class ModelLimits:
    """Configuration for model request limits."""

    request_frequency: int
    expiration_time_ms: int


class ThreadSafeTokenManager:
    """Thread-safe token management with proper synchronization."""

    def __init__(self, config: ConfigurationManager):
        """Initialize token manager with configuration."""
        self.config = config
        self._lock = threading.RLock()
        self._token_storage: Dict[str, List[TokenEntry]] = {}
        self._token_status: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._expired_tokens: List[Tuple[str, str, int, TokenType]] = []

        self._super_limits = {
            "grok-3": ModelLimits(50, 12 * 60 * 60 * 1000),
            "grok-4": ModelLimits(25, 12 * 60 * 60 * 1000),
            "grok-4-deepsearch": ModelLimits(10, 12 * 60 * 60 * 1000),
        }

        self._normal_limits = {
            "grok-3": ModelLimits(5, 12 * 60 * 60 * 1000),
            "grok-4": ModelLimits(5, 12 * 60 * 60 * 1000),
            "grok-4-deepsearch": ModelLimits(2, 12 * 60 * 60 * 1000),
        }

        self._reset_timer_started = False
        self._load_token_status()

        if not self._reset_timer_started and any(
            self._token_storage.get(m) for m in self._token_storage
        ):
            self._start_reset_timer()

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name for consistent lookup - all aliases map to base models."""
        if model.startswith("grok-"):
            parts = model.split("-")
            if len(parts) >= 2:
                base_model = f"{parts[0]}-{parts[1]}"
                if base_model in ["grok-3", "grok-4"]:
                    return base_model
        return model

    def _get_model_limits(self, token_type: TokenType) -> Dict[str, ModelLimits]:
        """Get model limits based on token type."""
        return (
            self._super_limits if token_type == TokenType.SUPER else self._normal_limits
        )

    def _save_token_status(self) -> None:
        """Save token status to persistent storage."""
        try:
            status_file = Path(self.config.get("TOKEN_STATUS_FILE"))
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump(self._token_status, f, indent=2, ensure_ascii=False)
            print("Token status saved to file")
        except Exception as e:
            print(f"Failed to save token status: {e}")

    def _load_token_status(self) -> None:
        """Load token status from persistent storage and reconstruct token storage."""
        try:
            status_file = Path(self.config.get("TOKEN_STATUS_FILE"))
            if status_file.exists():
                with open(status_file, "r", encoding="utf-8") as f:
                    self._token_status = json.load(f)
                print("Token status loaded from file")

                self._reconstruct_token_storage()
        except Exception as e:
            print(f"Failed to load token status: {e}")
            self._token_status = {}

    def _reconstruct_token_storage(self) -> None:
        """Reconstruct _token_storage from _token_status."""
        try:
            reconstructed_count = 0
            migration_count = 0

            for sso_value, models_data in self._token_status.items():
                is_super = False
                for model_data in models_data.values():
                    if model_data.get("isSuper", False):
                        is_super = True
                        break

                token_type = TokenType.SUPER if is_super else TokenType.NORMAL
                model_limits = self._get_model_limits(token_type)

                missing_models = []
                for model_name, limits in model_limits.items():
                    if model_name not in models_data:
                        missing_models.append(model_name)

                if missing_models:
                    for model_name in missing_models:
                        limits = model_limits[model_name]

                        models_data[model_name] = {
                            "isValid": True,
                            "invalidatedTime": None,
                            "totalRequestCount": 0,
                            "isSuper": is_super,
                            "max_request_count": limits.request_frequency,
                            "request_count": 0,
                            "added_time": int(time.time() * 1000),
                            "start_call_time": None,
                            "failed_request_count": 0,
                            "is_expired": False,
                            "last_failure_time": None,
                            "last_failure_reason": None,
                        }
                        migration_count += 1
                        print(
                            f"Added missing {model_name} data for token {sso_value[:20]}..."
                        )

                for model, model_data in models_data.items():
                    is_super = model_data.get("isSuper", False)
                    token_type = TokenType.SUPER if is_super else TokenType.NORMAL

                    credential = TokenCredential.from_raw_token(sso_value, token_type)

                    token_entry = TokenEntry(
                        credential=credential,
                        max_request_count=model_data.get("max_request_count", 20),
                        request_count=model_data.get("request_count", 0),
                        added_time=model_data.get(
                            "added_time", int(time.time() * 1000)
                        ),
                        start_call_time=model_data.get("start_call_time"),
                    )

                    if model_data.get("is_expired", False):
                        print(f"Skipping expired token for {model}")
                        continue

                    if model not in self._token_storage:
                        self._token_storage[model] = []

                    existing = next(
                        (
                            entry
                            for entry in self._token_storage[model]
                            if entry.credential.sso_token == credential.sso_token
                        ),
                        None,
                    )

                    if not existing:
                        self._token_storage[model].append(token_entry)
                        reconstructed_count += 1

            if reconstructed_count > 0:
                print(f"Reconstructed {reconstructed_count} token entries")

            if migration_count > 0:
                print(
                    f"Added missing model data to {migration_count} token-model combinations for backward compatibility"
                )
                self._save_token_status()

        except Exception as e:
            print(f"Failed to reconstruct token storage: {e}")

    def record_token_failure(
        self, model: str, token_string: str, failure_reason: str, status_code: int
    ) -> None:
        """Record a token failure and potentially mark as expired."""
        need_save = False

        with self._lock:
            try:
                normalized_model = self._normalize_model_name(model)
                credential = TokenCredential(token_string, TokenType.NORMAL)
                sso_value = credential.extract_sso_value()

                if (
                    sso_value in self._token_status
                    and normalized_model in self._token_status[sso_value]
                ):
                    status = self._token_status[sso_value][normalized_model]
                    status["failed_request_count"] = (
                        status.get("failed_request_count", 0) + 1
                    )
                    status["last_failure_time"] = int(time.time() * 1000)
                    status["last_failure_reason"] = f"{status_code}: {failure_reason}"

                    failure_threshold = 3
                    if status[
                        "failed_request_count"
                    ] >= failure_threshold and status_code in [401, 403]:
                        status["is_expired"] = True
                        status["isValid"] = False
                        print(
                            f"Token marked as expired after {status['failed_request_count']} failures: {failure_reason}",
                            "TokenManager",
                        )

                    need_save = True
                    print(
                        f"Recorded token failure for {model}: {failure_reason} (total failures: {status['failed_request_count']})",
                        "TokenManager",
                    )

            except Exception as e:
                print(f"Failed to record token failure: {e}")

        if need_save:
            try:
                self._save_token_status()
            except Exception as e:
                print(f"Failed to save token status: {e}")

    def _is_token_expired(self, token_entry: TokenEntry, model: str) -> bool:
        """Check if a token is marked as expired."""
        try:
            sso_value = token_entry.credential.extract_sso_value()
            if (
                sso_value in self._token_status
                and model in self._token_status[sso_value]
            ):
                status = self._token_status[sso_value][model]
                return status.get("is_expired", False)
            return False
        except Exception as e:
            print(f"Failed to check token expiration: {e}")
            return False

    def add_token(
        self, credential: TokenCredential, is_initialization: bool = False
    ) -> bool:
        """Add token to the management system."""
        need_save = False

        with self._lock:
            try:
                model_limits = self._get_model_limits(credential.token_type)
                sso_value = credential.extract_sso_value()

                for model, limits in model_limits.items():
                    if model not in self._token_storage:
                        self._token_storage[model] = []

                    existing_entry = next(
                        (
                            entry
                            for entry in self._token_storage[model]
                            if entry.credential.sso_token == credential.sso_token
                        ),
                        None,
                    )

                    if not existing_entry:
                        token_entry = TokenEntry(
                            credential=credential,
                            max_request_count=limits.request_frequency,
                            request_count=0,
                            added_time=int(time.time() * 1000),
                        )
                        self._token_storage[model].append(token_entry)

                        if sso_value not in self._token_status:
                            self._token_status[sso_value] = {}

                        if model not in self._token_status[sso_value]:
                            self._token_status[sso_value][model] = {
                                "isValid": True,
                                "invalidatedTime": None,
                                "totalRequestCount": 0,
                                "isSuper": credential.token_type == TokenType.SUPER,
                                "max_request_count": limits.request_frequency,
                                "request_count": 0,
                                "added_time": token_entry.added_time,
                                "start_call_time": None,
                                "failed_request_count": 0,
                                "is_expired": False,
                                "last_failure_time": None,
                                "last_failure_reason": None,
                            }

                if not is_initialization:
                    need_save = True

                print(
                    f"Token added successfully for type: {credential.token_type.value}",
                    "TokenManager",
                )

            except Exception as e:
                print(f"Failed to add token: {e}")
                return False

        if need_save:
            try:
                self._save_token_status()
            except Exception as e:
                print(f"Failed to save token status: {e}")

        return True

    def get_token_for_model(self, model: str) -> Optional[str]:
        """Get available token for specified model."""
        token_to_return = None
        need_save = False

        with self._lock:
            normalized_model = self._normalize_model_name(model)
            tokens = self._token_storage.get(normalized_model)
            if not tokens:
                return None

            for i, token_entry in enumerate(tokens):

                if self._is_token_expired(token_entry, normalized_model):
                    continue

                if token_entry.is_available():
                    token_entry.use_token()

                    try:
                        sso_value = token_entry.credential.extract_sso_value()
                        if (
                            sso_value in self._token_status
                            and normalized_model in self._token_status[sso_value]
                        ):
                            status = self._token_status[sso_value][normalized_model]
                            status["totalRequestCount"] += 1
                            status["request_count"] = token_entry.request_count
                            status["start_call_time"] = token_entry.start_call_time

                            if (
                                token_entry.request_count
                                >= token_entry.max_request_count
                            ):
                                status["isValid"] = False
                                status["invalidatedTime"] = int(time.time() * 1000)

                    except Exception as e:
                        print(f"Failed to update token status: {e}")

                    if not self._reset_timer_started:
                        self._start_reset_timer()

                    token_to_return = token_entry.credential.sso_token
                    need_save = True

                    if len(tokens) > 1:
                        tokens.append(tokens.pop(i))
                    break

            if token_to_return is None:
                now_ms = int(time.time() * 1000)
                for i, token_entry in enumerate(tokens):
                    limits = self._get_model_limits(
                        token_entry.credential.token_type
                    ).get(normalized_model)
                    if not limits:
                        continue

                    if token_entry.can_be_reset(limits.expiration_time_ms, now_ms):

                        token_entry.reset_usage()

                        try:
                            sso_value = token_entry.credential.extract_sso_value()
                            if (
                                sso_value in self._token_status
                                and normalized_model in self._token_status[sso_value]
                            ):
                                status = self._token_status[sso_value][normalized_model]
                                status["isValid"] = True
                                status["invalidatedTime"] = None
                                status["totalRequestCount"] = 0
                                status["request_count"] = 0
                                status["start_call_time"] = None
                        except Exception as e:
                            print(
                                f"Failed to opportunistically reset token status: {e}"
                            )

                        token_entry.use_token()
                        token_to_return = token_entry.credential.sso_token
                        need_save = True

                        try:
                            sso_value = token_entry.credential.extract_sso_value()
                            if (
                                sso_value in self._token_status
                                and normalized_model in self._token_status[sso_value]
                            ):
                                status = self._token_status[sso_value][normalized_model]
                                status["request_count"] = token_entry.request_count
                                status["start_call_time"] = token_entry.start_call_time
                        except Exception as e:
                            print(
                                f"Failed to update token status after opportunistic reset: {e}"
                            )

                        if len(tokens) > 1:
                            tokens.append(tokens.pop(i))

                        if not self._reset_timer_started:
                            self._start_reset_timer()
                        break

        if need_save:
            try:
                self._save_token_status()
            except Exception as e:
                print(f"Failed to save token status: {e}")

        return token_to_return

    def remove_token_from_model(self, model: str, token_string: str) -> bool:
        """Remove specific token from model permanently (will not be reactivated)."""
        with self._lock:
            normalized_model = self._normalize_model_name(model)

            if normalized_model not in self._token_storage:
                return False

            tokens = self._token_storage[normalized_model]
            for i, token_entry in enumerate(tokens):
                if token_entry.credential.sso_token == token_string:
                    tokens.pop(i)
                    print(f"Token permanently removed from model {model}")
                    return True

            return False

    def get_token_count_for_model(self, model: str) -> int:
        """Get schedulable (non-expired) token count for model."""
        with self._lock:
            normalized_model = self._normalize_model_name(model)
            tokens = self._token_storage.get(normalized_model, [])

            return sum(
                1
                for token in tokens
                if not self._is_token_expired(token, normalized_model)
            )

    def get_available_token_count(self, model: str) -> int:
        """Get immediately available (non-expired and not rate-limited) token count for model."""
        with self._lock:
            normalized_model = self._normalize_model_name(model)
            tokens = self._token_storage.get(normalized_model, [])

            return sum(
                1
                for token in tokens
                if not self._is_token_expired(token, normalized_model)
                and token.is_available()
            )

    def get_remaining_capacity(self) -> Dict[str, int]:
        """Get remaining request capacity for each model."""
        with self._lock:
            capacity_map = {}

            for model, tokens in self._token_storage.items():
                total_capacity = sum(entry.max_request_count for entry in tokens)
                used_requests = sum(entry.request_count for entry in tokens)
                capacity_map[model] = max(0, total_capacity - used_requests)

            return capacity_map

    def reduce_token_request_count(self, model: str, count: int) -> bool:
        """Reduce token request count (for error recovery)."""
        with self._lock:
            normalized_model = self._normalize_model_name(model)

            if normalized_model not in self._token_storage:
                return False

            tokens = self._token_storage[normalized_model]
            if not tokens:
                return False

            token_entry = tokens[0]
            original_count = token_entry.request_count
            token_entry.request_count = max(0, token_entry.request_count - count)
            reduction = original_count - token_entry.request_count

            try:
                sso_value = token_entry.credential.extract_sso_value()
                if (
                    sso_value in self._token_status
                    and normalized_model in self._token_status[sso_value]
                ):
                    status = self._token_status[sso_value][normalized_model]
                    status["totalRequestCount"] = max(
                        0, status["totalRequestCount"] - reduction
                    )
            except Exception as e:
                print(
                    f"Failed to update token status during reduction: {e}",
                    "TokenManager",
                )

            return True

    def _start_reset_timer(self) -> None:
        """Start the token reset timer."""

        def reset_expired_tokens():
            while True:
                try:
                    current_time = int(time.time() * 1000)
                    need_save = False

                    with self._lock:
                        tokens_to_remove = []
                        for token_info in self._expired_tokens:
                            token, model, expired_time, token_type = token_info
                            model_limits = self._get_model_limits(token_type)

                            if model in model_limits:
                                expiration_time = model_limits[model].expiration_time_ms

                                if current_time - expired_time >= expiration_time:
                                    try:
                                        credential = TokenCredential(token, token_type)
                                        self._reactivate_token(
                                            model, credential, model_limits[model]
                                        )
                                        tokens_to_remove.append(token_info)
                                    except Exception as e:
                                        print(
                                            f"Failed to reactivate token: {e}",
                                            "TokenManager",
                                        )

                        for token_info in tokens_to_remove:
                            self._expired_tokens.remove(token_info)

                        for model, tokens in self._token_storage.items():
                            for token_entry in tokens:
                                token_type = token_entry.credential.token_type
                                model_limits = self._get_model_limits(token_type)

                                if model in model_limits:
                                    if token_entry.can_be_reset(
                                        model_limits[model].expiration_time_ms,
                                        current_time,
                                    ):
                                        token_entry.reset_usage()
                                        need_save = True

                                        try:
                                            sso_value = (
                                                token_entry.credential.extract_sso_value()
                                            )
                                            if (
                                                sso_value in self._token_status
                                                and model
                                                in self._token_status[sso_value]
                                            ):
                                                status = self._token_status[sso_value][
                                                    model
                                                ]
                                                status["isValid"] = True
                                                status["invalidatedTime"] = None
                                                status["totalRequestCount"] = 0
                                                status["request_count"] = (
                                                    token_entry.request_count
                                                )
                                                status["start_call_time"] = (
                                                    token_entry.start_call_time
                                                )
                                        except Exception as e:
                                            print(
                                                f"Failed to update status during reset: {e}",
                                                "TokenManager",
                                            )

                    if need_save:
                        try:
                            self._save_token_status()
                        except Exception as e:
                            print(f"Failed to save token status during reset: {e}")

                except Exception as e:
                    print(f"Error in token reset timer: {e}")

                time.sleep(3600)

        timer_thread = threading.Thread(target=reset_expired_tokens, daemon=True)
        timer_thread.start()
        self._reset_timer_started = True

    def _reactivate_token(
        self, model: str, credential: TokenCredential, limits: ModelLimits
    ) -> None:
        """Reactivate an expired token."""
        existing = next(
            (
                entry
                for entry in self._token_storage.get(model, [])
                if entry.credential.sso_token == credential.sso_token
            ),
            None,
        )

        if not existing:
            if model not in self._token_storage:
                self._token_storage[model] = []

            token_entry = TokenEntry(
                credential=credential,
                max_request_count=limits.request_frequency,
                request_count=0,
                added_time=int(time.time() * 1000),
            )
            self._token_storage[model].append(token_entry)

            try:
                sso_value = credential.extract_sso_value()
                if sso_value in self._token_status:
                    if model not in self._token_status[sso_value]:
                        self._token_status[sso_value][model] = {}

                    status = self._token_status[sso_value][model]
                    status["isValid"] = True
                    status["invalidatedTime"] = None
                    status["totalRequestCount"] = 0
                    status["isSuper"] = credential.token_type == TokenType.SUPER
                    status["max_request_count"] = token_entry.max_request_count
                    status["request_count"] = token_entry.request_count
                    status["added_time"] = token_entry.added_time
                    status["start_call_time"] = token_entry.start_call_time
            except Exception as e:
                print(f"Failed to update reactivated token status: {e}")

    def delete_token(self, token_string: str) -> bool:
        """Delete token completely from the system."""
        removed = False

        with self._lock:
            try:
                credential = TokenCredential(token_string, TokenType.NORMAL)
                sso_value = credential.extract_sso_value()

                for model in self._token_storage:
                    tokens = self._token_storage[model]
                    original_length = len(tokens)
                    self._token_storage[model] = [
                        entry
                        for entry in tokens
                        if entry.credential.sso_token != token_string
                    ]
                    if len(self._token_storage[model]) < original_length:
                        removed = True

                if sso_value in self._token_status:
                    del self._token_status[sso_value]

                self._expired_tokens = [
                    token_info
                    for token_info in self._expired_tokens
                    if token_info[0] != token_string
                ]

            except Exception as e:
                print(f"Failed to delete token: {e}")
                return False

        if removed:
            try:
                self._save_token_status()
                print(f"Token deleted successfully")
            except Exception as e:
                print(f"Failed to save token status after deletion: {e}")

        return removed

    def get_all_tokens(self) -> List[str]:
        """Get all token strings in the system."""
        with self._lock:
            all_tokens = set()
            for tokens in self._token_storage.values():
                for entry in tokens:
                    all_tokens.add(entry.credential.sso_token)
            return list(all_tokens)

    def get_token_status_map(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get complete token status mapping."""
        with self._lock:
            return dict(self._token_status)

    def get_token_health_summary(self) -> Dict[str, Any]:
        """Get summary of token health across all models."""
        with self._lock:
            summary = {
                "total_tokens": 0,
                "healthy_tokens": 0,
                "expired_tokens": 0,
                "rate_limited_tokens": 0,
                "tokens_with_failures": 0,
                "total_failures": 0,
                "by_model": {},
            }

            unique_tokens = set()
            token_health_status = {}

            for sso_value, models_data in self._token_status.items():
                unique_tokens.add(sso_value)

                is_expired = False
                is_rate_limited = False
                has_failures = False
                total_token_failures = 0

                for model, model_data in models_data.items():

                    if model not in summary["by_model"]:
                        summary["by_model"][model] = {
                            "total": 0,
                            "healthy": 0,
                            "expired": 0,
                            "rate_limited": 0,
                            "with_failures": 0,
                        }

                    summary["by_model"][model]["total"] += 1

                    if model_data.get("is_expired", False):
                        summary["by_model"][model]["expired"] += 1
                        is_expired = True
                    elif not model_data.get("isValid", True):
                        summary["by_model"][model]["rate_limited"] += 1
                        is_rate_limited = True
                    else:
                        summary["by_model"][model]["healthy"] += 1

                    failure_count = model_data.get("failed_request_count", 0)
                    if failure_count > 0:
                        summary["by_model"][model]["with_failures"] += 1
                        has_failures = True
                        total_token_failures += failure_count

                token_health_status[sso_value] = {
                    "is_expired": is_expired,
                    "is_rate_limited": is_rate_limited,
                    "has_failures": has_failures,
                    "total_failures": total_token_failures,
                }

            summary["total_tokens"] = len(unique_tokens)

            for sso_value in unique_tokens:
                status = token_health_status[sso_value]

                if status["is_expired"]:
                    summary["expired_tokens"] += 1
                elif status["is_rate_limited"]:
                    summary["rate_limited_tokens"] += 1
                else:
                    summary["healthy_tokens"] += 1

                if status["has_failures"]:
                    summary["tokens_with_failures"] += 1
                    summary["total_failures"] += status["total_failures"]

            return summary


@dataclass
class ImageTypeInfo:
    """Information about an image type."""

    mime_type: str
    file_name: str
    extension: str


class ImageProcessor:
    """Handles image processing and type detection."""

    IMAGE_SIGNATURES = {
        b"\xff\xd8\xff": ("jpg", "image/jpeg"),
        b"\x89PNG\r\n\x1a\n": ("png", "image/png"),
        b"GIF89a": ("gif", "image/gif"),
        b"GIF87a": ("gif", "image/gif"),
    }

    @classmethod
    def is_base64_image(cls, s: str) -> bool:
        """Check if string is a valid base64 image by examining binary signatures."""
        try:
            decoded = base64.b64decode(s, validate=True)
            return any(decoded.startswith(sig) for sig in cls.IMAGE_SIGNATURES)
        except Exception:
            return False

    @classmethod
    def get_extension_and_mime_from_header(cls, data: bytes) -> tuple:
        """Detect image format from binary header."""
        for sig, (ext, mime) in cls.IMAGE_SIGNATURES.items():
            if data.startswith(sig):
                return ext, mime
        return "jpg", "image/jpeg"

    @classmethod
    def get_image_type_info(cls, base64_string: str) -> ImageTypeInfo:
        """Enhanced image type detection with binary signature support."""
        mime_type = "image/jpeg"
        extension = "jpg"

        if "data:image" in base64_string:
            matches = re.search(
                r"data:([a-zA-Z0-9]+\/[a-zA-Z0-9-.+]+);base64,", base64_string
            )
            if matches:
                mime_type = matches.group(1)
                extension = mime_type.split("/")[1]
        else:
            try:
                image_data = base64.b64decode(base64_string, validate=True)
                extension, mime_type = cls.get_extension_and_mime_from_header(
                    image_data
                )
            except Exception:
                pass

        file_name = f"image.{extension}"
        return ImageTypeInfo(mime_type, file_name, extension)


class FileUploadManager:
    """Handles file and image upload operations."""

    def __init__(
        self, config: ConfigurationManager, token_manager: ThreadSafeTokenManager
    ):
        """Initialize file upload manager."""
        self.config = config
        self.token_manager = token_manager

    def upload_text_file(self, content: str, model: str) -> str:
        """Upload text content as a file attachment."""
        try:
            content_base64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
            upload_data = {
                "fileName": "message.txt",
                "fileMimeType": "text/plain",
                "content": content_base64,
            }

            print("Uploading text file")

            auth_token = self.token_manager.get_token_for_model(model)
            if not auth_token:
                raise TokenException(f"No available tokens for model: {model}")

            cf_clearance = self.config.get("SERVER.CF_CLEARANCE", "")
            cookie = f"{auth_token};{cf_clearance}" if cf_clearance else auth_token

            proxy_config = UtilityFunctions.get_proxy_configuration(
                self.config.get("API.PROXY")
            )

            response = curl_requests.post(
                "https://grok.com/rest/app-chat/upload-file",
                headers={
                    **get_dynamic_headers(
                        "POST", "/rest/app-chat/upload-file", self.config
                    ),
                    "Cookie": cookie,
                },
                json=upload_data,
                impersonate="chrome133a",
                timeout=60,
                **proxy_config,
            )

            if response.status_code != 200:
                raise GrokApiException(
                    f"File upload failed with status: {response.status_code}",
                    "UPLOAD_FAILED",
                )

            result = response.json()
            file_metadata_id = result.get("fileMetadataId", "")

            if not file_metadata_id:
                raise GrokApiException(
                    "No file metadata ID in response", "INVALID_RESPONSE"
                )

            print(f"Text file uploaded successfully: {file_metadata_id}", "FileUpload")
            return file_metadata_id

        except Exception as error:
            print(f"Text file upload failed: {error}")
            raise GrokApiException(
                f"Text file upload failed: {error}", "UPLOAD_ERROR"
            ) from error

    def upload_image(self, image_data: str, model: str) -> str:
        """Upload image with enhanced format support."""
        try:
            if "data:image" in image_data:
                image_buffer = image_data.split(",")[1]
            else:
                image_buffer = image_data

            image_info = ImageProcessor.get_image_type_info(image_data)

            upload_data = {
                "fileName": image_info.file_name,
                "fileMimeType": image_info.mime_type,
                "content": image_buffer,
            }

            print("Uploading image file")

            auth_token = self.token_manager.get_token_for_model(model)
            if not auth_token:
                raise TokenException(f"No available tokens for model: {model}")

            cf_clearance = self.config.get("SERVER.CF_CLEARANCE", "")
            cookie = f"{auth_token};{cf_clearance}" if cf_clearance else auth_token

            proxy_config = UtilityFunctions.get_proxy_configuration(
                self.config.get("API.PROXY")
            )

            response = curl_requests.post(
                "https://grok.com/rest/app-chat/upload-file",
                headers={
                    **get_dynamic_headers(
                        "POST", "/rest/app-chat/upload-file", self.config
                    ),
                    "Cookie": cookie,
                },
                json=upload_data,
                impersonate="chrome133a",
                timeout=60,
                **proxy_config,
            )

            if response.status_code != 200:
                print(
                    f"Image upload failed with status: {response.status_code}",
                    "ImageUpload",
                )
                return ""

            result = response.json()
            file_metadata_id = result.get("fileMetadataId", "")

            if file_metadata_id:
                print(f"Image uploaded successfully: {file_metadata_id}", "ImageUpload")

            return file_metadata_id

        except Exception as error:
            print(f"Image upload failed: {error}")
            return ""


@dataclass
class ProcessedMessage:
    """Result of message processing."""

    content: str
    file_attachments: List[str]
    requires_file_upload: bool
    upload_content: str = ""


class MessageContentProcessor:
    """Processes message content and handles complex formats."""

    def __init__(self, file_upload_manager: FileUploadManager):
        """Initialize message processor."""
        self.file_upload_manager = file_upload_manager

    def remove_think_tags_and_images(self, text: str) -> str:
        """Remove think tags and base64 images from text."""
        text = re.sub(r"<think>[\s\S]*?<\/think>", "", text).strip()
        text = re.sub(r"!\[image\]\(data:.*?base64,.*?\)", "[图片]", text)
        return text

    def process_content_item(self, content_item: Any) -> str:
        """Process individual content item (text or image)."""
        if isinstance(content_item, list):
            text_parts = []
            for item in content_item:
                if isinstance(item, dict):
                    if item.get("type") == "image_url":
                        text_parts.append("[图片]")
                    elif item.get("type") == "text":
                        text_parts.append(
                            self.remove_think_tags_and_images(item.get("text", ""))
                        )
            return "\n".join(filter(None, text_parts))

        elif isinstance(content_item, dict):
            if content_item.get("type") == "image_url":
                return "[图片]"
            elif content_item.get("type") == "text":
                return self.remove_think_tags_and_images(content_item.get("text", ""))

        elif isinstance(content_item, str):
            return self.remove_think_tags_and_images(content_item)

        return ""

    def extract_image_attachments(self, content_item: Any, model: str) -> List[str]:
        """Extract and upload image attachments from content."""
        attachments = []

        if isinstance(content_item, list):
            for item in content_item:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url:
                        file_id = self.file_upload_manager.upload_image(
                            image_url, model
                        )
                        if file_id:
                            attachments.append(file_id)

        elif isinstance(content_item, dict) and content_item.get("type") == "image_url":
            image_url = content_item.get("image_url", {}).get("url", "")
            if image_url:
                file_id = self.file_upload_manager.upload_image(image_url, model)
                if file_id:
                    attachments.append(file_id)

        return attachments

    def process_messages(
        self, messages: List[Dict[str, Any]], model: str
    ) -> ProcessedMessage:
        """Process list of messages into a single formatted string."""
        message_lines = []
        all_file_attachments = []
        message_length = 0
        requires_file_upload = False
        last_role = None
        last_content = ""

        for message in messages:
            role = "assistant" if message.get("role") == "assistant" else "user"
            is_last_message = message == messages[-1]

            if is_last_message and "content" in message:
                image_attachments = self.extract_image_attachments(
                    message["content"], model
                )
                all_file_attachments.extend(image_attachments)

            text_content = self.process_content_item(message.get("content", ""))

            if text_content or (is_last_message and all_file_attachments):
                if role == last_role and text_content and message_lines:

                    last_content += "\n" + text_content

                    message_lines[-1] = f"{role.upper()}: {last_content}"
                    newly_added = "\n" + text_content
                else:
                    content_to_add = text_content or "[图片]"
                    line = f"{role.upper()}: {content_to_add}"
                    message_lines.append(line)
                    last_content = text_content
                    last_role = role
                    newly_added = line

                message_length += len(newly_added)

            if message_length >= MESSAGE_LENGTH_LIMIT:
                requires_file_upload = True

        formatted_messages = "\n".join(message_lines) + "\n" if message_lines else ""

        if requires_file_upload:
            last_message = messages[-1] if messages else {}
            last_role = (
                "assistant" if last_message.get("role") == "assistant" else "user"
            )
            last_text = self.process_content_item(last_message.get("content", ""))

            final_content = f"{last_role.upper()}: {last_text or '[图片]'}"

            try:
                file_id = self.file_upload_manager.upload_text_file(
                    formatted_messages, model
                )
                if file_id:
                    all_file_attachments.insert(0, file_id)
                    formatted_messages = "基于txt文件内容进行回复："
            except Exception as e:
                print(f"Failed to upload conversation file: {e}", "MessageProcessor")
                formatted_messages = final_content

        if not formatted_messages.strip():
            if requires_file_upload:
                formatted_messages = "基于txt文件内容进行回复："
            else:
                raise ValidationException("Message content is empty after processing")

        return ProcessedMessage(
            content=formatted_messages.strip(),
            file_attachments=all_file_attachments[:MAX_FILE_ATTACHMENTS],
            requires_file_upload=requires_file_upload,
        )


@dataclass
class ChatRequestConfig:
    """Configuration for chat request with new architecture."""

    model_name: str
    message: str
    file_attachments: List[str]
    enable_search: bool
    enable_image_generation: bool
    enable_reasoning: bool
    deepsearch_preset: str
    temporary_conversation: bool

    model_mode: str = "MODEL_MODE_AUTO"
    workspace_ids: List[str] = None  # type: ignore
    token_pool: str = "grok-3"

    def __post_init__(self):
        if self.workspace_ids is None:
            self.workspace_ids = []


class GrokApiClient:
    """Clean, focused Grok API client with separated concerns."""

    def __init__(
        self, config: ConfigurationManager, token_manager: ThreadSafeTokenManager
    ):
        """Initialize Grok API client."""
        self.config = config
        self.token_manager = token_manager
        self.file_upload_manager = FileUploadManager(config, token_manager)
        self.message_processor = MessageContentProcessor(self.file_upload_manager)

    def validate_model_and_request(
        self, model: str, request_data: Dict[str, Any]
    ) -> str:
        """Validate model and request parameters."""
        if model not in self.config.models:
            raise ValidationException(f"Unsupported model: {model}")

        return self.config.models[model]

    def determine_request_settings(
        self, requested_model: str, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine request settings based on model alias and request data."""

        settings = {
            "base_model": "grok-3",
            "model_mode": "MODEL_MODE_AUTO",
            "enable_search": False,
            "enable_image_generation": False,
            "enable_reasoning": False,
            "workspace_ids": [],
            "token_pool": "grok-3",
        }

        if request_data.get("search"):
            settings["enable_search"] = True
        if request_data.get("image") or request_data.get("enableImageGeneration"):
            settings["enable_image_generation"] = True
        if request_data.get("reasoning") or request_data.get("isReasoning"):
            settings["enable_reasoning"] = True

        model_alias = requested_model.lower()

        if model_alias in ["grok-3", "grok-auto"]:
            settings["base_model"] = "grok-3"
            settings["model_mode"] = "MODEL_MODE_AUTO"
            settings["token_pool"] = "grok-3"

        elif model_alias == "grok-fast":
            settings["base_model"] = "grok-3"
            settings["model_mode"] = "MODEL_MODE_FAST"
            settings["token_pool"] = "grok-3"

        elif model_alias in ["grok-4", "grok-expert"]:
            settings["base_model"] = "grok-4"
            settings["model_mode"] = "MODEL_MODE_EXPERT"
            settings["token_pool"] = "grok-4"

        elif model_alias in ["grok-search", "grok-deepsearch"]:
            settings["base_model"] = "grok-4"
            settings["model_mode"] = "MODEL_MODE_EXPERT"
            settings["enable_search"] = True
            settings["workspace_ids"] = [str(uuid.uuid4())]
            settings["token_pool"] = "grok-4-deepsearch"

        elif model_alias in ["grok-image", "grok-draw"]:
            settings["base_model"] = "grok-4"
            settings["model_mode"] = "MODEL_MODE_EXPERT"
            settings["enable_image_generation"] = True
            settings["token_pool"] = "grok-4"

        if request_data.get("model_mode"):
            settings["model_mode"] = request_data["model_mode"]
        if request_data.get("workspace_ids"):
            settings["workspace_ids"] = request_data["workspace_ids"]

        return settings

    def validate_message_requirements(
        self, model: str, messages: List[Dict[str, Any]]
    ) -> None:
        """Validate message requirements for specific models."""
        if model in ["grok-4-imageGen", "grok-3-imageGen", "grok-3-deepsearch"]:
            if not messages:
                raise ValidationException("Messages cannot be empty")

            last_message = messages[-1]
            if last_message.get("role") != "user":
                raise ValidationException(
                    f"Model {model} requires the last message to be from user"
                )

    def prepare_chat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare chat request with new architecture."""
        try:
            model = str(request_data.get("model"))
            messages = request_data.get("messages", [])

            normalized_model = self.validate_model_and_request(model, request_data)

            settings = self.determine_request_settings(model, request_data)

            self.validate_message_requirements(model, messages)

            if model in ["grok-image", "grok-draw"]:
                messages = [messages[-1]]

            processed_message = self.message_processor.process_messages(
                messages, normalized_model
            )

            request_config = ChatRequestConfig(
                model_name=settings["base_model"],
                message=processed_message.content,
                file_attachments=processed_message.file_attachments,
                enable_search=settings["enable_search"],
                enable_image_generation=settings["enable_image_generation"],
                enable_reasoning=settings["enable_reasoning"],
                deepsearch_preset="",
                temporary_conversation=self.config.get(
                    "API.IS_TEMP_CONVERSATION", False
                ),
                model_mode=settings["model_mode"],
                workspace_ids=settings["workspace_ids"],
                token_pool=settings["token_pool"],
            )

            return self.build_request_payload(request_config)

        except Exception as e:
            print(f"Failed to prepare chat request: {e}")
            raise

    def build_request_payload(self, config: ChatRequestConfig) -> Dict[str, Any]:
        """Build the final request payload with new architecture."""
        payload = {
            "temporary": config.temporary_conversation,
            "modelName": config.model_name,
            "message": config.message,
            "fileAttachments": config.file_attachments,
            "imageAttachments": [],
            "disableSearch": not config.enable_search,
            "enableImageGeneration": config.enable_image_generation,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": True,
            "imageGenerationCount": 2,
            "forceConcise": False,
            "toolOverrides": {},
            "enableSideBySide": True,
            "sendFinalMetadata": True,
            "isReasoning": config.enable_reasoning,
            "webpageUrls": [],
            "disableTextFollowUps": True,
            "responseMetadata": {"requestModelDetails": {"modelId": config.model_name}},
            "disableMemory": False,
            "forceSideBySide": False,
            "modelMode": config.model_mode,
            "isAsyncChat": False,
            "tokenPool": config.token_pool,
        }

        if config.workspace_ids:
            payload["workspaceIds"] = config.workspace_ids

        return payload

    def make_request(
        self, payload: Dict[str, Any], model: str, stream: bool = False
    ) -> Tuple[requests.Response, str]:
        """Make the actual HTTP request to Grok API and return response with used token."""

        token_pool = payload.get("tokenPool", payload.get("modelName", model))

        auth_token = self.token_manager.get_token_for_model(token_pool)
        if not auth_token:
            active_count = self.token_manager.get_token_count_for_model(token_pool)
            available_count = self.token_manager.get_available_token_count(token_pool)

            if active_count == 0:
                raise TokenException(
                    f"No tokens available for model: {token_pool}. Please add tokens or check configuration."
                )
            elif available_count == 0:
                raise TokenException(
                    f"All tokens for model {token_pool} are currently rate limited. Please try again later."
                )
            else:

                raise TokenException(
                    f"Unable to obtain token for model {token_pool}. Please try again later."
                )

        cf_clearance = self.config.get("SERVER.CF_CLEARANCE", "")
        cookie = f"{auth_token};{cf_clearance}" if cf_clearance else auth_token

        proxy_config = UtilityFunctions.get_proxy_configuration(
            self.config.get("API.PROXY")
        )

        print(
            f"Making request to Grok API for model: {model} (using token pool: {token_pool})"
        )

        try:
            response = curl_requests.post(
                f"{self.config.get('API.BASE_URL')}/rest/app-chat/conversations/new",
                headers={
                    **get_dynamic_headers(
                        "POST", "/rest/app-chat/conversations/new", self.config
                    ),
                    "Cookie": cookie,
                },
                data=json.dumps(payload),
                impersonate="chrome133a",
                stream=stream,
                timeout=120,
                **proxy_config,
            )

            print(f"Response status: {response.status_code}")
            return response, auth_token  # type: ignore

        except Exception as e:
            print(f"HTTP request failed: {e}")
            raise GrokApiException(f"HTTP request failed: {e}", "REQUEST_FAILED") from e


@dataclass
class ProcessingResult:
    """Result of processing a model response."""

    token: Optional[str] = None
    image_url: Optional[str] = None
    new_state: Optional[ProcessingState] = None
    should_skip: bool = False


class ModelResponseProcessor:
    """Stateless response processor for different model types."""

    def __init__(self, config: ConfigurationManager):
        """Initialize response processor."""
        self.config = config

    def process_response(
        self, response_data: Dict[str, Any], model: str, current_state: ProcessingState
    ) -> ProcessingResult:
        """Process model response based on model type and current state."""
        try:
            streaming_image_response = response_data.get(
                "streamingImageGenerationResponse"
            )
            if streaming_image_response:
                progress = streaming_image_response.get("progress", 0)
                image_url = streaming_image_response.get("imageUrl")

                if progress == 100 and image_url:
                    new_state = current_state.with_image_generation(True, 1)
                    return ProcessingResult(image_url=image_url, new_state=new_state)
                else:
                    new_state = current_state.with_image_generation(True)
                    return ProcessingResult(new_state=new_state)

            if response_data.get("doImgGen") or response_data.get(
                "imageAttachmentInfo"
            ):
                new_state = current_state.with_image_generation(True)
                return ProcessingResult(new_state=new_state)

            if current_state.is_generating_image:
                cached_response = response_data.get("cachedImageGenerationResponse")
                if cached_response and not current_state.image_generation_phase:
                    image_url = cached_response.get("imageUrl")
                    if image_url:
                        new_state = current_state.with_image_generation(True, 1)
                        return ProcessingResult(
                            image_url=image_url, new_state=new_state
                        )

            model_response = response_data.get("modelResponse")
            if model_response:
                generated_image_urls = model_response.get("generatedImageUrls", [])
                if generated_image_urls:
                    image_url = generated_image_urls[0]
                    new_state = current_state.with_image_generation(True, 1)
                    return ProcessingResult(image_url=image_url, new_state=new_state)

            if model == "grok-3":
                return self._process_grok_3_response(response_data, current_state)
            elif model == "grok-3-search":
                return self._process_grok_3_search_response(
                    response_data, current_state
                )
            elif model in [
                "grok-3-deepsearch",
                "grok-3-deepersearch",
                "grok-4-deepsearch",
            ]:
                return self._process_deepsearch_response(response_data, current_state)
            elif model == "grok-3-reasoning":
                return self._process_reasoning_response(response_data, current_state)
            elif model == "grok-4":
                return self._process_grok_4_response(response_data, current_state)
            elif model == "grok-4-reasoning":
                return self._process_grok_4_reasoning_response(
                    response_data, current_state
                )
            else:
                token = response_data.get("token")
                processed_token = self._transform_artifacts(token) if token else None
                return ProcessingResult(token=processed_token)

        except Exception as e:
            print(f"Error processing {model} response: {e}", "ResponseProcessor")
            token = response_data.get("token")
            processed_token = self._transform_artifacts(token) if token else None
            return ProcessingResult(token=processed_token)

    def _process_grok_3_response(
        self, response_data: Dict[str, Any], current_state: ProcessingState
    ) -> ProcessingResult:
        """Process Grok-3 model response."""
        token = response_data.get("token")
        processed_token = self._transform_artifacts(token) if token else None
        return ProcessingResult(token=processed_token, new_state=current_state)

    def _process_grok_3_search_response(
        self, response_data: Dict[str, Any], current_state: ProcessingState
    ) -> ProcessingResult:
        """Process Grok-3 search model response."""
        if response_data.get("webSearchResults") and self.config.get(
            "SHOW_SEARCH_RESULTS", True
        ):
            search_results = UtilityFunctions.organize_search_results(
                response_data["webSearchResults"]
            )
            token = f"\r\n<think>{search_results}</think>\r\n"
            processed_token = self._transform_artifacts(token)
            return ProcessingResult(token=processed_token, new_state=current_state)
        else:
            token = response_data.get("token")
            processed_token = self._transform_artifacts(token) if token else None
            return ProcessingResult(token=processed_token, new_state=current_state)

    def _process_deepsearch_response(
        self, response_data: Dict[str, Any], current_state: ProcessingState
    ) -> ProcessingResult:
        """Process deep search model responses with thinking state management."""
        show_thinking = self.config.get("SHOW_THINKING", False)
        has_step_id = bool(response_data.get("messageStepId"))
        message_tag = response_data.get("messageTag", "")

        if has_step_id and not show_thinking:
            return ProcessingResult(should_skip=True, new_state=current_state)

        if has_step_id and not current_state.is_thinking:
            token = "<think>" + response_data.get("token", "")
            processed_token = self._transform_artifacts(token)
            new_state = current_state.with_thinking(True)
            return ProcessingResult(token=processed_token, new_state=new_state)

        if not has_step_id and current_state.is_thinking and message_tag == "final":
            token = "</think>" + response_data.get("token", "")
            processed_token = self._transform_artifacts(token)
            new_state = current_state.with_thinking(False)
            return ProcessingResult(token=processed_token, new_state=new_state)

        if (
            has_step_id and current_state.is_thinking and message_tag == "assistant"
        ) or message_tag == "final":
            token = response_data.get("token", "")
            processed_token = self._transform_artifacts(token) if token else None
            return ProcessingResult(token=processed_token, new_state=current_state)

        if current_state.is_thinking and isinstance(response_data.get("token"), dict):
            token_dict = response_data.get("token", {})
            if token_dict.get("action") == "webSearch":
                action_input = token_dict.get("action_input", {})
                query = action_input.get("query", "")
                processed_token = self._transform_artifacts(query) if query else None
                return ProcessingResult(token=processed_token, new_state=current_state)

        if current_state.is_thinking and response_data.get("webSearchResults"):
            search_results = UtilityFunctions.organize_search_results(
                response_data["webSearchResults"]
            )
            processed_token = self._transform_artifacts(search_results)
            return ProcessingResult(token=processed_token, new_state=current_state)

        return ProcessingResult(new_state=current_state)

    def _process_reasoning_response(
        self, response_data: Dict[str, Any], current_state: ProcessingState
    ) -> ProcessingResult:
        """Process reasoning model responses."""
        show_thinking = self.config.get("SHOW_THINKING", False)
        is_thinking = response_data.get("isThinking", False)

        if is_thinking and not show_thinking:
            return ProcessingResult(should_skip=True, new_state=current_state)

        if is_thinking and not current_state.is_thinking:
            token = "<think>" + response_data.get("token", "")
            processed_token = self._transform_artifacts(token)
            new_state = current_state.with_thinking(True)
            return ProcessingResult(token=processed_token, new_state=new_state)

        if not is_thinking and current_state.is_thinking:
            token = "</think>" + response_data.get("token", "")
            processed_token = self._transform_artifacts(token)
            new_state = current_state.with_thinking(False)
            return ProcessingResult(token=processed_token, new_state=new_state)

        token = response_data.get("token")
        processed_token = self._transform_artifacts(token) if token else None
        return ProcessingResult(token=processed_token, new_state=current_state)

    def _process_grok_4_response(
        self, response_data: Dict[str, Any], current_state: ProcessingState
    ) -> ProcessingResult:
        """Process Grok-4 model response."""
        if response_data.get("isThinking"):
            return ProcessingResult(should_skip=True, new_state=current_state)

        token = response_data.get("token")
        processed_token = self._transform_artifacts(token) if token else None
        return ProcessingResult(token=processed_token, new_state=current_state)

    def _process_grok_4_reasoning_response(
        self, response_data: Dict[str, Any], current_state: ProcessingState
    ) -> ProcessingResult:
        """Process Grok-4 reasoning model response."""
        show_thinking = self.config.get("SHOW_THINKING", False)
        is_thinking = response_data.get("isThinking", False)
        message_tag = response_data.get("messageTag", "")

        if is_thinking and not show_thinking:
            return ProcessingResult(should_skip=True, new_state=current_state)

        if is_thinking and not current_state.is_thinking and message_tag == "assistant":
            token = "<think>" + response_data.get("token", "")
            processed_token = self._transform_artifacts(token)
            new_state = current_state.with_thinking(True)
            return ProcessingResult(token=processed_token, new_state=new_state)

        if not is_thinking and current_state.is_thinking and message_tag == "final":
            token = "</think>" + response_data.get("token", "")
            processed_token = self._transform_artifacts(token)
            new_state = current_state.with_thinking(False)
            return ProcessingResult(token=processed_token, new_state=new_state)

        token = response_data.get("token")
        processed_token = self._transform_artifacts(token) if token else None
        return ProcessingResult(token=processed_token, new_state=current_state)

    def _transform_artifacts(self, text: Any) -> str:
        """Artifact transformation now handled at streaming level - return unchanged."""
        if not text:
            return ""

        return str(text) if not isinstance(text, str) else text


class ResponseImageHandler:
    """Handles image responses with memory caching and OpenAI-compatible format."""

    def __init__(self, config: ConfigurationManager):
        """Initialize image handler with memory cache."""
        self.config = config
        self._cache = {}
        self._cache_lock = threading.Lock()
        self.max_cache_items = 128
        self.cache_access_order = []

    def handle_image_response(self, image_url: str) -> dict:
        """Process image response and return OpenAI-compatible format with caching."""
        with self._cache_lock:
            if image_url in self._cache:
                self.cache_access_order.remove(image_url)
                self.cache_access_order.append(image_url)
                return self._cache[image_url]

        max_retries = 2
        retry_count = 0
        image_data = None

        while retry_count < max_retries:
            try:
                proxy_config = UtilityFunctions.get_proxy_configuration(
                    self.config.get("API.PROXY")
                )

                response = curl_requests.get(
                    f"https://assets.grok.com/{image_url}",
                    headers=get_dynamic_headers(
                        "GET", f"/assets/{image_url}", self.config
                    ),
                    impersonate="chrome133a",
                    timeout=60,
                    **proxy_config,
                )

                if response.status_code == 200:
                    image_data = response.content
                    break

                retry_count += 1
                if retry_count == max_retries:
                    raise GrokApiException(
                        f"Failed to retrieve image after {max_retries} attempts: {response.status_code}",
                        "IMAGE_RETRIEVAL_FAILED",
                    )

                time.sleep(self.config.get("API.RETRY_TIME", 1000) / 1000 * retry_count)

            except Exception as error:
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Image retrieval failed: {error}")
                    raise

                time.sleep(self.config.get("API.RETRY_TIME", 1000) / 1000 * retry_count)

        if not image_data:
            raise GrokApiException("No image data retrieved", "NO_IMAGE_DATA")

        base64_image = base64.b64encode(image_data).decode("utf-8")
        ext, content_type = ImageProcessor.get_extension_and_mime_from_header(
            image_data
        )
        data_url = f"data:{content_type};base64,{base64_image}"

        image_content = {"type": "image_url", "image_url": {"url": data_url}}

        with self._cache_lock:
            if (
                len(self._cache) >= self.max_cache_items
                and image_url not in self._cache
            ):
                oldest_key = self.cache_access_order.pop(0)
                del self._cache[oldest_key]

            self._cache[image_url] = image_content
            if image_url not in self.cache_access_order:
                self.cache_access_order.append(image_url)

        return image_content


from enum import Enum


class FilterState(Enum):
    """States for the streaming tag filter state machine."""

    NORMAL = "normal"
    POTENTIAL_TAG = "potential_tag"
    IN_FILTERED_TAG = "in_filtered_tag"
    IN_PRESERVED_TAG = "in_preserved_tag"
    IN_CDATA = "in_cdata"
    TAG_ANALYSIS = "tag_analysis"


class TagBehavior(Enum):
    """Behavior types for filtered tags."""

    PRESERVE_CONTENT = "preserve_content"
    REMOVE_ALL = "remove_all"


class StreamingTagFilter:
    """High-performance state machine-based streaming filter with minimal buffering and per-stream independence."""

    def __init__(
        self,
        tag_config: Dict[str, Dict[str, Any]] = {},
        content_type_mappings: Dict[str, Dict[str, str]] = {},
    ):
        """
        Initialize filter with configurable tag behaviors and contentType mappings.

        Args:
            tag_config: Dict mapping tag names to their behavior configuration
                       e.g., {
                           "xaiartifact": {"behavior": "preserve_content"},
                           "grok:render": {"behavior": "remove_all"}
                       }
            content_type_mappings: Dict mapping contentTypes to replacement tags
        """
        self.tag_config = {}
        default_config = tag_config or {
            "xaiartifact": {"behavior": "preserve_content"},
            "grok:render": {"behavior": "remove_all"},
        }

        for tag_name, config in default_config.items():
            self.tag_config[tag_name.lower()] = {
                "behavior": TagBehavior(config.get("behavior", "preserve_content")),
                "extra_config": config.get("extra_config", {}),
            }
        self.content_type_mappings = content_type_mappings or {
            "text/plain": {"stag": "```", "etag": "```"},
            "text/markdown": {"stag": "", "etag": ""},
            "application/json": {"stag": "```json\n", "etag": "\n```"},
            "text/javascript": {"stag": "```javascript\n", "etag": "\n```"},
            "text/python": {"stag": "```python\n", "etag": "\n```"},
            "text/html": {"stag": "```html\n", "etag": "\n```"},
            "text/css": {"stag": "```css\n", "etag": "\n```"},
            "text/xml": {"stag": "```xml\n", "etag": "\n```"},
            "text/yaml": {"stag": "```yaml\n", "etag": "\n```"},
            "text/sql": {"stag": "```sql\n", "etag": "\n```"},
            "text/typescript": {"stag": "```typescript\n", "etag": "\n```"},
            "text/bash": {"stag": "```bash\n", "etag": "\n```"},
            "text/shell": {"stag": "```bash\n", "etag": "\n```"},
            "text/dockerfile": {"stag": "```dockerfile\n", "etag": "\n```"},
            "text/java": {"stag": "```java\n", "etag": "\n```"},
            "text/go": {"stag": "```go\n", "etag": "\n```"},
            "text/rust": {"stag": "```rust\n", "etag": "\n```"},
            "text/php": {"stag": "```php\n", "etag": "\n```"},
            "text/ruby": {"stag": "```ruby\n", "etag": "\n```"},
            "text/swift": {"stag": "```swift\n", "etag": "\n```"},
            "text/kotlin": {"stag": "```kotlin\n", "etag": "\n```"},
            "text/cpp": {"stag": "```cpp\n", "etag": "\n```"},
            "text/c": {"stag": "```c\n", "etag": "\n```"},
            "text/csharp": {"stag": "```csharp\n", "etag": "\n```"},
            "text/code": {"stag": "```\n", "etag": "\n```"},
            "application/code": {"stag": "```\n", "etag": "\n```"},
        }

        self.reset_state()

    def _get_tag_behavior(self, tag_name: str) -> Optional[TagBehavior]:
        """Get the configured behavior for a tag."""
        config = self.tag_config.get(tag_name.lower())
        return config["behavior"] if config else None

    def _is_filtered_tag(self, tag_name: str) -> bool:
        """Check if a tag is configured for filtering."""
        return tag_name.lower() in self.tag_config

    def reset_state(self):
        """Reset the filter state (useful for reusing filter instances)."""
        self.state = FilterState.NORMAL
        self.buffer = ""
        self.tag_stack = []
        self.temp_output = ""
        self.has_mismatched_closing_tags = False

        self.has_filtered_tags_in_text = False
        self.last_char_was_lt = False

    def _quick_scan_for_filtered_content(self, text: str) -> bool:
        """Quick scan to see if text potentially contains filtered content."""
        if not text or "<" not in text:
            return False

        text_lower = text.lower()

        if "<![cdata[" in text_lower:
            return True

        for tag_name in self.tag_config.keys():
            if f"<{tag_name}" in text_lower or f"</{tag_name}" in text_lower:
                return True

        return False

    def _extract_tag_name_quick(self, tag_content: str) -> str:
        """Quick tag name extraction for performance."""
        if not tag_content:
            return ""

        if tag_content.startswith("/"):
            tag_content = tag_content[1:]

        if tag_content.endswith("/"):
            tag_content = tag_content[:-1]

        parts = tag_content.split(None, 1)
        return parts[0].lower() if parts else ""

    def _extract_content_type(self, tag_content: str) -> Optional[str]:
        """Extract contentType attribute from tag."""
        match = re.search(
            r'contentType=["\']([^"\'>]+)["\']', tag_content, re.IGNORECASE
        )
        return match.group(1) if match else None

    def _should_preserve_content(
        self, tag_name: str, content_type: Optional[str]
    ) -> bool:
        """Check if tag content should be preserved with replacement."""
        behavior = self._get_tag_behavior(tag_name)
        return behavior == TagBehavior.PRESERVE_CONTENT

    def _get_content_replacement(self, content_type: Optional[str]) -> Dict[str, str]:
        """Get replacement mapping for content type, with fallback to plain text."""
        if content_type and content_type in self.content_type_mappings:
            return self.content_type_mappings[content_type]
        return {"stag": "", "etag": ""}

    def _process_complete_tag(self, tag_text: str) -> str:
        """Process a complete tag and return appropriate output."""
        if not tag_text.startswith("<") or not tag_text.endswith(">"):
            return tag_text

        tag_content = tag_text[1:-1]

        if tag_content.lower().startswith("![cdata["):
            return ""

        tag_name = self._extract_tag_name_quick(tag_content)
        behavior = self._get_tag_behavior(tag_name)

        if behavior is None:
            return tag_text

        is_closing = tag_content.startswith("/")
        is_self_closing = tag_content.endswith("/")

        if is_closing:
            matched = False
            for i in range(len(self.tag_stack) - 1, -1, -1):
                if self.tag_stack[i]["name"] == tag_name:
                    tag_entry = self.tag_stack.pop(i)
                    self.tag_stack = self.tag_stack[:i]
                    matched = True

                    if tag_entry.get("preserve_content"):
                        return tag_entry["replacement"].get("etag", "")
                    break

            if not matched:
                if self._is_in_preserved_context():
                    self.has_mismatched_closing_tags = True
                    return tag_text
            return ""

        elif is_self_closing:
            if behavior == TagBehavior.PRESERVE_CONTENT:
                content_type = self._extract_content_type(tag_content)
                if self._should_preserve_content(tag_name, content_type):
                    replacement = self._get_content_replacement(content_type)
                    return replacement.get("stag", "") + replacement.get("etag", "")
            return ""

        else:
            content_type = self._extract_content_type(tag_content)
            preserve_content = self._should_preserve_content(tag_name, content_type)

            replacement = (
                self._get_content_replacement(content_type) if preserve_content else {}
            )

            self.tag_stack.append(
                {
                    "name": tag_name,
                    "behavior": behavior,
                    "content_type": content_type,
                    "preserve_content": preserve_content,
                    "replacement": replacement,
                }
            )

            if preserve_content:
                return replacement.get("stag", "")
            return ""

    def _might_be_closing_tag(self, tag_text: str) -> bool:
        """Check if the tag might be a closing tag for any tag in our stack."""
        if not tag_text.startswith("</"):
            return False

        tag_content = tag_text[1:-1] if tag_text.endswith(">") else tag_text[1:]
        tag_name = self._extract_tag_name_quick(tag_content)

        for tag_entry in self.tag_stack:
            if tag_entry["name"] == tag_name:
                return True
        return False

    def _is_in_removal_context(self) -> bool:
        """Check if we're currently in a removal context.
        ANY REMOVE_ALL tag in the stack means we're in removal context."""
        for tag_entry in self.tag_stack:
            if tag_entry.get("behavior") == TagBehavior.REMOVE_ALL:
                return True
        return False

    def _is_in_preserved_context(self) -> bool:
        """Check if we're currently in a content-preserving context.
        The most recent (top of stack) tag's behavior takes precedence."""
        if not self.tag_stack:
            return False
        top_tag = self.tag_stack[-1]
        return top_tag.get("preserve_content", False)

    def _is_in_filtered_context(self) -> bool:
        """Check if we're currently in any filtered context."""
        return len(self.tag_stack) > 0

    def _should_output_char(self, char: str) -> bool:
        """Determine if a character should be output based on current context."""
        if not self._is_in_filtered_context():
            return True
        if self._is_in_removal_context():
            return False
        return self._is_in_preserved_context()

    def filter_chunk(self, chunk: str) -> str:
        """Filter chunk with minimal buffering and maximum streaming efficiency."""
        if not chunk:
            return ""

        if (
            self.state == FilterState.NORMAL
            and not self.tag_stack
            and not self.buffer
            and not self._quick_scan_for_filtered_content(chunk)
            and "<" not in chunk
        ):
            return chunk

        result = ""
        i = 0

        while i < len(chunk):
            char = chunk[i]

            if self.state == FilterState.NORMAL:
                if char == "<":
                    self.state = FilterState.POTENTIAL_TAG
                    self.buffer = "<"
                else:
                    if self._should_output_char(char):
                        result += char
                i += 1

            elif self.state == FilterState.POTENTIAL_TAG:
                self.buffer += char

                if char == ">":
                    tag_output = self._process_complete_tag(self.buffer)

                    if not self._is_in_removal_context():
                        result += tag_output

                    self.buffer = ""
                    self.state = FilterState.NORMAL
                    i += 1

                elif len(self.buffer) > 1 and self.buffer.lower().startswith(
                    "<![cdata["
                ):
                    self.state = FilterState.IN_CDATA
                    i += 1

                elif len(self.buffer) > 256:
                    first_char = self.buffer[0]
                    if self._should_output_char(first_char):
                        result += first_char

                    self.buffer = self.buffer[1:]

                    if not self.buffer or not self.buffer.startswith("<"):
                        if self.buffer:
                            for buf_char in self.buffer:
                                if self._should_output_char(buf_char):
                                    result += buf_char
                        self.buffer = ""
                        self.state = FilterState.NORMAL
                    continue
                else:
                    i += 1

            elif self.state == FilterState.IN_CDATA:
                self.buffer += char
                if self.buffer.endswith("]]>"):
                    self.buffer = ""
                    self.state = FilterState.NORMAL
                i += 1

        return result

    def finalize(self) -> str:
        """Finalize the stream and return any remaining content."""
        result = ""

        if self.buffer:
            if self.state == FilterState.POTENTIAL_TAG:
                buffer_lower = self.buffer.lower()
                is_filtered = False

                for tag_name in self.tag_config.keys():
                    if buffer_lower.startswith(
                        f"<{tag_name}"
                    ) or buffer_lower.startswith(f"</{tag_name}"):
                        is_filtered = True
                        break

                if not is_filtered and self._should_output_char(self.buffer[0]):
                    result += self.buffer
            elif self.state == FilterState.NORMAL and self._should_output_char(
                self.buffer[0]
            ):
                result += self.buffer

        if not self.has_mismatched_closing_tags:
            for tag_entry in reversed(self.tag_stack):
                if tag_entry.get("preserve_content"):
                    if tag_entry.get("replacement"):
                        result += tag_entry["replacement"].get("etag", "")

        self.reset_state()

        return result


@dataclass
class StreamingContext:
    """Context for streaming response processing."""

    model: str
    processor: ModelResponseProcessor
    image_handler: ResponseImageHandler
    tag_filter: StreamingTagFilter
    state: ProcessingState = field(default_factory=ProcessingState)


class StreamProcessor:
    """Handles streaming response processing."""

    @staticmethod
    def process_non_stream_response(
        response: requests.Response, context: StreamingContext
    ) -> Union[str, Dict[str, Any]]:
        """Process non-streaming response and return complete content or image response."""
        print("Processing non-streaming response")

        full_response = ""
        current_state = context.state

        try:
            for chunk in response.iter_lines():
                if not chunk:
                    continue

                try:
                    line_data = json.loads(chunk.decode("utf-8").strip())

                    if line_data.get("error"):
                        error_info = line_data.get("error", {})
                        error_message = error_info.get("message", "Unknown error")
                        print(f"API error: {json.dumps(line_data, indent=2)}")
                        return f"Error: {error_message}"

                    response_data = line_data.get("result", {}).get("response")
                    if not response_data:
                        continue

                    result = context.processor.process_response(
                        response_data, context.model, current_state
                    )

                    if result.new_state:
                        current_state = result.new_state

                    if result.should_skip:
                        continue

                    if result.token:
                        filtered_token = context.tag_filter.filter_chunk(result.token)
                        if filtered_token:
                            full_response += filtered_token

                    if result.image_url:
                        image_content = context.image_handler.handle_image_response(
                            result.image_url
                        )
                        return MessageProcessor.create_chat_completion_with_content(
                            image_content, context.model
                        )

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing response line: {e}")
                    continue

            final_content = context.tag_filter.finalize()
            if final_content:
                full_response += final_content

            return full_response

        except Exception as e:
            print(f"Non-stream response processing failed: {e}")
            raise

    @staticmethod
    def process_stream_response(
        response: requests.Response, context: StreamingContext
    ) -> Iterator[str]:
        """Process streaming response and yield formatted chunks."""
        print("Processing streaming response")

        current_state = context.state

        try:
            for chunk in response.iter_lines():
                if not chunk:
                    continue

                try:
                    line_data = json.loads(chunk.decode("utf-8").strip())

                    if line_data.get("error"):
                        print(f"API error: {json.dumps(line_data, indent=2)}")
                        yield "data: " + json.dumps(
                            {"error": "RateLimitError"}
                        ) + "\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    response_data = line_data.get("result", {}).get("response")
                    if not response_data:
                        continue

                    result = context.processor.process_response(
                        response_data, context.model, current_state
                    )

                    if result.new_state:
                        current_state = result.new_state

                    if result.should_skip:
                        continue

                    if result.token:
                        filtered_token = context.tag_filter.filter_chunk(result.token)

                        if filtered_token:
                            formatted_response = (
                                MessageProcessor.create_chat_completion_chunk(
                                    filtered_token, context.model
                                )
                            )
                            yield f"data: {json.dumps(formatted_response)}\n\n"

                    if result.image_url:
                        image_content = context.image_handler.handle_image_response(
                            result.image_url
                        )
                        formatted_response = (
                            MessageProcessor.create_chat_completion_chunk_with_content(
                                image_content, context.model
                            )
                        )
                        yield f"data: {json.dumps(formatted_response)}\n\n"

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing stream line: {e}")
                    continue

            final_content = context.tag_filter.finalize()
            if final_content:
                formatted_response = MessageProcessor.create_chat_completion_chunk(
                    final_content, context.model
                )
                yield f"data: {json.dumps(formatted_response)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"Stream processing failed: {e}")
            error_response = MessageProcessor.create_error_response(str(e))
            yield f"data: {json.dumps(error_response)}\n\n"


class MessageProcessor:
    """Creates properly formatted chat completion responses."""

    @staticmethod
    def create_chat_completion(message: str, model: str) -> Dict[str, Any]:
        """Create a complete chat completion response."""
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": message},
                    "finish_reason": "stop",
                }
            ],
            "usage": None,
        }

    @staticmethod
    def create_chat_completion_with_content(content: Any, model: str) -> Dict[str, Any]:
        """Create a complete chat completion response with content array support."""
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": None,
        }

    @staticmethod
    def create_chat_completion_chunk(message: str, model: str) -> Dict[str, Any]:
        """Create a streaming chat completion chunk."""
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": message}}],
        }

    @staticmethod
    def create_chat_completion_chunk_with_content(
        content: Any, model: str
    ) -> Dict[str, Any]:
        """Create a streaming chat completion chunk with content array support."""
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content}}],
        }

    @staticmethod
    def create_error_response(error_message: str) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "error": {"message": error_message, "type": "server_error"},
        }


class AuthenticationService:
    """Handles authentication and authorization."""

    def __init__(
        self, config: ConfigurationManager, token_manager: ThreadSafeTokenManager
    ):
        """Initialize authentication service."""
        self.config = config
        self.token_manager = token_manager

    def validate_api_key(self, auth_header: Optional[str]) -> str:
        """Validate API key from authorization header."""
        if not auth_header:
            raise ValidationException("Authorization header missing")

        auth_token = auth_header.replace("Bearer ", "").strip()
        if not auth_token:
            raise ValidationException("API key missing")

        return auth_token

    def process_authentication(self, auth_header: Optional[str]) -> bool:
        """Process authentication and add token if needed."""
        try:
            auth_token = self.validate_api_key(auth_header)

            if self.config.get("API.IS_CUSTOM_SSO", False):
                try:
                    credential = TokenCredential.from_raw_token(
                        auth_token, TokenType.NORMAL
                    )
                    success = self.token_manager.add_token(credential)
                    if not success:
                        print("Failed to add custom SSO token")
                    return True
                except Exception as e:
                    print(
                        f"Failed to process custom SSO token: {e}",
                        "AuthenticationService",
                    )
                    raise ValidationException(f"Invalid SSO token format: {e}")
            else:
                expected_key = self.config.get("API.API_KEY", "sk-123456")
                if auth_token != expected_key:
                    print(f"Invalid API key provided")
                    raise ValidationException("Invalid API key")
                return True

        except ValidationException:
            raise
        except Exception as e:
            print(f"Authentication processing failed: {e}")
            raise ValidationException(f"Authentication failed: {e}") from e


def create_app(
    config: ConfigurationManager, token_manager: ThreadSafeTokenManager
) -> Flask:
    """Create and configure Flask application."""
    app = Flask(__name__)

    app.config["SECRET_KEY"] = secrets.token_urlsafe(32)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    auth_service = AuthenticationService(config, token_manager)
    grok_client = GrokApiClient(config, token_manager)
    response_processor = ModelResponseProcessor(config)
    image_handler = ResponseImageHandler(config)

    def create_error_response(
        error_data: Union[str, Dict[str, Any]], status_code: int
    ) -> Tuple[Dict[str, Any], int]:
        """Create consistent error responses."""
        return UtilityFunctions.create_structured_error_response(
            error_data, status_code
        )

    @app.errorhandler(ValidationException)
    def handle_validation_error(e: ValidationException) -> Tuple[Dict[str, Any], int]:
        """Handle validation exceptions."""
        return create_error_response(
            {"error": str(e), "error_code": "VALIDATION_ERROR"}, 400
        )

    @app.errorhandler(TokenException)
    def handle_token_error(e: TokenException) -> Tuple[Dict[str, Any], int]:
        """Handle token-related exceptions."""
        return create_error_response(
            {"error": str(e), "error_code": "TOKEN_ERROR"}, 429
        )

    @app.errorhandler(RateLimitException)
    def handle_rate_limit_error(e: RateLimitException) -> Tuple[Dict[str, Any], int]:
        """Handle rate limiting exceptions."""
        return create_error_response(
            {"error": str(e), "error_code": "RATE_LIMIT_ERROR"}, 429
        )

    @app.errorhandler(GrokApiException)
    def handle_grok_api_error(e: GrokApiException) -> Tuple[Dict[str, Any], int]:
        """Handle Grok API exceptions."""
        return create_error_response({"error": str(e), "error_code": e.error_code}, 500)

    @app.errorhandler(500)
    def handle_internal_error(e) -> Tuple[Dict[str, Any], int]:
        """Handle internal server errors."""
        print(f"Internal server error: {e}")
        return create_error_response("Internal server error", 500)

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions():
        """Main chat completions endpoint with clean separation of concerns."""
        response_status_code = None

        try:
            if not request.is_json:
                raise ValidationException("Request must be JSON")

            data = request.get_json()
            if not data:
                raise ValidationException("Request body is empty")

            auth_service.process_authentication(request.headers.get("Authorization"))

            model = data.get("model")
            messages = data.get("messages", [])
            stream = data.get("stream", False)

            if not model:
                raise ValidationException("Model parameter is required")

            if not messages:
                raise ValidationException("Messages parameter is required")

            print(f"Processing chat completion request for model: {model}")

            payload = grok_client.prepare_chat_request(data)

            response = None
            used_token = None
            retry_count = 0
            max_retries = config.get("RETRY.MAX_ATTEMPTS", MAX_RETRY_ATTEMPTS)

            while retry_count < max_retries:
                try:
                    response, used_token = grok_client.make_request(
                        payload, model, stream
                    )
                    response_status_code = response.status_code

                    if response.status_code == 200:
                        break
                    elif response.status_code == 429:
                        if token_manager.get_available_token_count(model) > 0:
                            print("Rate limited, retrying with different token")
                            retry_count += 1
                            continue
                        else:
                            raise RateLimitException(
                                "Rate limit exceeded and no alternative tokens available"
                            )
                    else:
                        error_text = (
                            response.text
                            if response.text
                            else f"HTTP {response.status_code} error"
                        )
                        print(
                            f"API request failed with status {response.status_code}: {error_text}"
                        )
                        print(f"Full response headers: {dict(response.headers)}")
                        print(f"Full response content: {error_text[:2000]}")

                        if used_token:
                            token_manager.record_token_failure(
                                model, used_token, error_text, response.status_code
                            )

                        error_data = UtilityFunctions.parse_error_response(error_text)
                        raise GrokApiException(
                            error_data.get(
                                "error",
                                f"API request failed with status {response.status_code}",
                            ),
                            error_data.get("error_code", "API_ERROR"),
                        )

                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise GrokApiException(
                            f"Request failed after {max_retries} attempts: {e}",
                            "REQUEST_FAILED",
                        )

                    delay = BASE_RETRY_DELAY * (2 ** (retry_count - 1))
                    print(f"Request failed, retrying in {delay}s: {e}")
                    time.sleep(delay)

            if not response:
                error_msg = "Request failed: No response received"
                raise GrokApiException(error_msg, "REQUEST_FAILED")
            elif response.status_code != 200:
                if response_status_code:
                    error_msg = f"Request failed with status: {response_status_code}"
                else:
                    error_msg = f"Request failed with status: {response.status_code}"
                raise GrokApiException(error_msg, "REQUEST_FAILED")

            tag_config = config.get("TAG_CONFIG", {})
            content_type_mappings = config.get("CONTENT_TYPE_MAPPINGS", {})

            context = StreamingContext(
                model=model,
                processor=response_processor,
                image_handler=image_handler,
                tag_filter=StreamingTagFilter(tag_config, content_type_mappings),
            )

            if stream:

                def generate():
                    try:
                        yield from StreamProcessor.process_stream_response(
                            response, context
                        )
                    except Exception as e:
                        print(f"Stream processing error: {e}")
                        error_response = MessageProcessor.create_error_response(str(e))
                        yield f"data: {json.dumps(error_response)}\n\n"

                return Response(
                    stream_with_context(generate()),
                    content_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                response_result = StreamProcessor.process_non_stream_response(
                    response, context
                )
                if isinstance(response_result, dict):
                    formatted_response = response_result
                else:
                    formatted_response = MessageProcessor.create_chat_completion(
                        response_result, model
                    )
                return jsonify(formatted_response)

        except (
            ValidationException,
            TokenException,
            RateLimitException,
            GrokApiException,
        ):
            raise
        except Exception as e:
            print(f"Unexpected error in chat completions: {e}")
            raise GrokApiException("Internal server error", "INTERNAL_ERROR") from e

    @app.route("/v1/models", methods=["GET"])
    def list_models():
        """List available models."""
        models_data = []
        current_time = int(time.time())

        for model_key in config.models.keys():
            models_data.append(
                {
                    "id": model_key,
                    "object": "model",
                    "created": current_time,
                    "owned_by": "grok",
                }
            )

        return jsonify({"object": "list", "data": models_data})

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        return jsonify(
            {"status": "healthy", "timestamp": int(time.time()), "version": "2.0.0"}
        )

    @app.route("/", methods=["GET"])
    def index():
        """Index page with basic information."""
        return jsonify(
            {"message": "Grok API Gateway", "version": "2.0.0", "status": "running"}
        )

    def check_admin_auth() -> bool:
        """Check admin authentication."""
        if not config.get("ADMIN.MANAGER_SWITCH"):
            return False

        password = request.form.get("password") or request.args.get("password")
        expected_password = config.get("ADMIN.PASSWORD")

        return bool(password and expected_password and password == expected_password)

    @app.route("/add_token", methods=["POST"])
    def add_token():
        """Add token endpoint (admin only)."""
        if not check_admin_auth():
            return jsonify({"error": "Unauthorized"}), 401

        try:
            token_data = request.form.get("tokens") or (
                request.json and request.json.get("tokens")
            )
            if not token_data:
                return jsonify({"error": "Token data required"}), 400

            if isinstance(token_data, str):
                try:
                    token_dict = json.loads(token_data)
                except json.JSONDecodeError:
                    token_dict = {"token": token_data, "type": "normal"}
            else:
                token_dict = token_data

            token_string = token_dict.get("token", "")
            token_type_str = token_dict.get("type")
            token_type = (
                TokenType.SUPER if token_type_str == "super" else TokenType.NORMAL
            )

            if not token_string:
                return jsonify({"error": "Token string required"}), 400

            credential = TokenCredential(token_string, token_type)
            success = token_manager.add_token(credential)

            if success:
                return jsonify({"message": "Token added successfully"})
            else:
                return jsonify({"error": "Failed to add token"}), 500

        except Exception as e:
            print(f"Error adding token: {e}")
            return jsonify({"error": f"Failed to add token: {e}"}), 500

    @app.route("/tokens_info", methods=["GET"])
    def tokens_info():
        """Get token information (admin only)."""
        if not check_admin_auth():
            return jsonify({"error": "Unauthorized"}), 401

        try:
            status_map = token_manager.get_token_status_map()
            capacity_map = token_manager.get_remaining_capacity()
            health_summary = token_manager.get_token_health_summary()

            return jsonify(
                {
                    "token_status": status_map,
                    "remaining_capacity": capacity_map,
                    "health_summary": health_summary,
                }
            )
        except Exception as e:
            print(f"Error getting token info: {e}")
            return jsonify({"error": f"Failed to get token info: {e}"}), 500

    def check_session_auth() -> bool:
        """Check session-based authentication for web manager."""
        return session.get("is_logged_in", False)

    @app.route("/manager/login", methods=["GET", "POST"])
    def manager_login():
        """Manager login page and handler."""
        if not config.get("ADMIN.MANAGER_SWITCH"):
            return redirect("/")

        if request.method == "POST":
            password = request.form.get("password")
            if password == config.get("ADMIN.PASSWORD"):
                session["is_logged_in"] = True
                return redirect("/manager")
            return render_template("login.html", error=True)

        return render_template("login.html", error=False)

    @app.route("/manager")
    def manager():
        """Main manager dashboard."""
        if not check_session_auth():
            return redirect("/manager/login")
        return render_template("manager.html")

    @app.route("/manager/api/get")
    def get_manager_tokens():
        """Get tokens via manager API."""
        if not check_session_auth():
            return jsonify({"error": "Unauthorized"}), 401

        try:
            status_map = token_manager.get_token_status_map()
            health_summary = token_manager.get_token_health_summary()

            return jsonify(
                {"token_status": status_map, "health_summary": health_summary}
            )
        except Exception as e:
            print(f"Error getting manager tokens: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/manager/api/add", methods=["POST"])
    def add_manager_token():
        """Add token via manager API."""
        if not check_session_auth():
            return jsonify({"error": "Unauthorized"}), 401

        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "JSON data required"}), 400

            sso = data.get("sso")
            if not sso or not sso.strip():
                return (
                    jsonify({"error": "SSO token is required and cannot be empty"}),
                    400,
                )

            credential = TokenCredential.from_raw_token(sso.strip(), TokenType.NORMAL)
            success = token_manager.add_token(credential)

            if success:
                return jsonify({"success": True})
            else:
                return jsonify({"error": "Failed to add token"}), 500

        except Exception as e:
            print(f"Error adding manager token: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/manager/api/delete", methods=["POST"])
    def delete_manager_token():
        """Delete token via manager API."""
        if not check_session_auth():
            return jsonify({"error": "Unauthorized"}), 401

        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "JSON data required"}), 400

            sso = data.get("sso")
            if not sso:
                return jsonify({"error": "SSO token is required"}), 400

            token_string = f"sso-rw={sso};sso={sso}"
            success = token_manager.delete_token(token_string)

            if success:
                return jsonify({"success": True})
            else:
                return jsonify({"error": "Token not found or failed to delete"}), 404

        except Exception as e:
            print(f"Error deleting manager token: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/manager/api/cf_clearance", methods=["POST"])
    def set_cf_clearance():
        """Set CF clearance via manager API."""
        if not check_session_auth():
            return jsonify({"error": "Unauthorized"}), 401

        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "JSON data required"}), 400

            cf_clearance = data.get("cf_clearance")
            if not cf_clearance:
                return jsonify({"error": "cf_clearance is required"}), 400

            config.set("SERVER.CF_CLEARANCE", cf_clearance)
            return jsonify({"success": True})

        except Exception as e:
            print(f"Error setting CF clearance: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/get/tokens", methods=["GET"])
    def get_tokens():
        """Legacy endpoint to get tokens."""
        auth_token = request.headers.get("Authorization", "").replace("Bearer ", "")

        if config.get("API.IS_CUSTOM_SSO", False):
            return (
                jsonify(
                    {"error": "Custom SSO mode cannot get polling SSO token status"}
                ),
                403,
            )
        elif auth_token != config.get("API.API_KEY", "sk-123456"):
            return jsonify({"error": "Unauthorized"}), 401

        try:
            return jsonify(token_manager.get_token_status_map())
        except Exception as e:
            print(f"Error getting tokens: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/add/token", methods=["POST"])
    def add_token_api():
        """API endpoint to add tokens (API key auth)."""
        auth_token = request.headers.get("Authorization", "").replace("Bearer ", "")

        if config.get("API.IS_CUSTOM_SSO", False):
            return jsonify({"error": "Custom SSO mode cannot add SSO tokens"}), 403
        elif auth_token != config.get("API.API_KEY", "sk-123456"):
            return jsonify({"error": "Unauthorized"}), 401

        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "JSON data required"}), 400

            sso = data.get("sso")
            if not sso or not sso.strip():
                return (
                    jsonify({"error": "SSO token is required and cannot be empty"}),
                    400,
                )

            credential = TokenCredential.from_raw_token(sso.strip(), TokenType.NORMAL)
            success = token_manager.add_token(credential)

            if success:
                return jsonify({"message": "Token added successfully"})
            else:
                return jsonify({"error": "Failed to add token"}), 500

        except Exception as e:
            print(f"Error adding token via API: {e}")
            return jsonify({"error": str(e)}), 500

    return app


def initialize_application(
    config: ConfigurationManager, token_manager: ThreadSafeTokenManager
) -> None:
    """Initialize the application with environment tokens."""
    tokens_added = 0

    sso_tokens = os.environ.get("SSO", "")
    if sso_tokens:
        for token in sso_tokens.split(","):
            token = token.strip()
            if token:
                try:
                    credential = TokenCredential(token, TokenType.NORMAL)
                    if token_manager.add_token(credential, is_initialization=True):
                        tokens_added += 1
                except Exception as e:
                    print(f"Failed to add normal token: {e}")

    sso_super_tokens = os.environ.get("SSO_SUPER", "")
    if sso_super_tokens:
        for token in sso_super_tokens.split(","):
            token = token.strip()
            if token:
                try:
                    credential = TokenCredential(token, TokenType.SUPER)
                    if token_manager.add_token(credential, is_initialization=True):
                        tokens_added += 1
                except Exception as e:
                    print(f"Failed to add super token: {e}")

    if tokens_added > 0:
        print(f"Successfully loaded {tokens_added} tokens")
    else:
        print("No tokens loaded during initialization")
        if not config.get("API.IS_CUSTOM_SSO", False):
            print(
                "Set SSO or SSO_SUPER environment variables, or enable IS_CUSTOM_SSO",
                "Initialization",
            )

    print("Dynamic headers disabled - skipping StatsigManager initialization")

    print("Application initialization completed")


def cleanup_resources():
    """Clean up browser resources before shutdown"""
    pass


def main():
    """Main application entry point."""
    try:
        config = ConfigurationManager()
        token_manager = ThreadSafeTokenManager(config)

        initialize_application(config, token_manager)

        try:
            token_manager._save_token_status()
        except Exception as e:
            print(f"Warning: Failed to save token status during initialization: {e}")

        app = create_app(config, token_manager)

        port = config.get("SERVER.PORT", 5200)
        print(f"Starting Grok API Gateway on port {port}")

        import atexit

        atexit.register(cleanup_resources)

        app.run(
            host="0.0.0.0",
            port=port,
            debug=False,
            threaded=False,
            processes=1,
        )

    except KeyboardInterrupt:
        print("Application stopped by user")
        cleanup_resources()
    except Exception as e:
        print(f"Application failed to start: {e}")
        cleanup_resources()
        sys.exit(1)
    finally:
        cleanup_resources()


if __name__ == "__main__":
    main()
