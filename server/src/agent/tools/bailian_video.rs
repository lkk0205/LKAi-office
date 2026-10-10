use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::time::{sleep, Instant};

use crate::agent::tool::ToolContext;

/// 一次远程视频生成的统一结果（百炼 / Agnes 通用）
#[derive(Debug, Clone)]
pub struct RemoteVideo {
    pub video_url: String,
    pub seconds: String,
    pub size: String,
}

/// 调用阿里云百炼·万相（wan3.0-video）生成视频：创建异步任务并轮询至结束。
///
/// - 无 `first_frame`：文生视频，ratio 用用户宽高比
/// - 有 `first_frame`：首帧生视频，ratio=adaptive
pub async fn run_bailian_video(
    ctx: &ToolContext,
    prompt: &str,
    aspect_ratio: &str,
    duration_secs: u32,
    first_frame: Option<&str>,
) -> Result<RemoteVideo> {
    let config = crate::config::config();
    let base = config.bailian_video_base_url.trim().trim_end_matches('/');
    let api_key = config.bailian_video_api_key.trim();
    let model = config.bailian_video_model.trim();
    if base.is_empty() {
        return Err(anyhow!("未配置 BAILIAN_VIDEO_BASE_URL"));
    }
    if api_key.is_empty() {
        return Err(anyhow!("未配置 BAILIAN_VIDEO_API_KEY"));
    }

    let mut input = json!({ "prompt": prompt });
    let ratio = match first_frame {
        Some(frame) if !frame.is_empty() => {
            input["media"] = json!([{ "type": "first_frame", "url": frame }]);
            "adaptive"
        }
        _ => aspect_ratio,
    };
    let body = json!({
        "model": model,
        "input": input,
        "parameters": {
            "resolution": "480P",
            "ratio": ratio,
            "duration": duration_secs,
            // 提示词已由规划环节写得很详细，关闭百炼提示词扩展，避免其扩展服务 500
            "prompt_extend": false
        }
    });

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?;
    let create_url = format!("{base}/services/aigc/video-generation/video-synthesis");

    ctx.send(
        "state_update",
        json!({
            "phase": "running",
            "step": "提交百炼任务",
            "detail": format!("正在向阿里云百炼·万相提交视频任务（{aspect_ratio} / {duration_secs}秒）..."),
            "at": chrono::Utc::now().to_rfc3339(),
        }),
    );

    let resp = client
        .post(&create_url)
        .header("X-DashScope-Async", "enable")
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(anyhow!("百炼任务创建失败：{status}: {text}"));
    }
    let created: Value = serde_json::from_str(&text)?;
    let task_id = created
        .get("output")
        .and_then(|o| o.get("task_id"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("百炼未返回 task_id: {text}"))?
        .to_string();

    // 轮询任务结果；连续 3 次查询失败 / 超过 180 秒则判定百炼不可用，交给上层降级
    // 视频生成通常 1-5 分钟，给足 7 分钟（含排队）；任务 24h 内可查
    let deadline = Instant::now() + Duration::from_secs(420);
    let mut consecutive_errors = 0u32;

    loop {
        if Instant::now() >= deadline {
            return Err(anyhow!("百炼视频生成超时（task_id: {task_id}）"));
        }

        let qurl = format!("{base}/tasks/{task_id}");
        let qresp = match client
            .get(&qurl)
            .header("Authorization", format!("Bearer {api_key}"))
            .send()
            .await
        {
            Ok(r) => r,
            Err(_) => {
                consecutive_errors += 1;
                if consecutive_errors >= 3 {
                    return Err(anyhow!("百炼结果查询连续失败（task_id: {task_id}）"));
                }
                sleep(Duration::from_secs(3)).await;
                continue;
            }
        };

        let qstatus = qresp.status();
        let qtext = qresp.text().await.unwrap_or_default();
        if !qstatus.is_success() {
            consecutive_errors += 1;
            if consecutive_errors >= 3 {
                return Err(anyhow!("百炼结果查询返回 {qstatus}: {qtext}"));
            }
            sleep(Duration::from_secs(3)).await;
            continue;
        }
        consecutive_errors = 0;

        let qv: Value = serde_json::from_str(&qtext).unwrap_or(Value::Null);
        let output = qv.get("output");
        let task_status = output
            .and_then(|o| o.get("task_status"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let progress = output
            .and_then(|o| o.get("task_progress"))
            .and_then(Value::as_i64)
            .unwrap_or(0);

        ctx.send(
            "state_update",
            json!({
                "phase": "running",
                "step": "百炼视频生成中",
                "detail": format!("百炼·万相状态：{task_status}（{progress}%）"),
                "at": chrono::Utc::now().to_rfc3339(),
            }),
        );

        match task_status {
            "SUCCEEDED" => {
                let video_url = output
                    .and_then(|o| o.get("video_url"))
                    .and_then(Value::as_str)
                    .filter(|s| !s.trim().is_empty())
                    .ok_or_else(|| anyhow!("百炼任务成功但缺少 video_url"))?;
                let seconds = output
                    .and_then(|o| o.get("duration"))
                    .map(Value::to_string)
                    .unwrap_or_else(|| duration_secs.to_string());
                return Ok(RemoteVideo {
                    video_url: video_url.to_string(),
                    seconds,
                    size: "480P".to_string(),
                });
            }
            "FAILED" => {
                let msg = output
                    .and_then(|o| o.get("message"))
                    .or_else(|| output.and_then(|o| o.get("code")))
                    .map(Value::to_string)
                    .unwrap_or_else(|| "未知错误".to_string());
                return Err(anyhow!("百炼视频生成失败：{msg}"));
            }
            "CANCELED" => return Err(anyhow!("百炼任务被取消")),
            _ => sleep(Duration::from_secs(3)).await,
        }
    }
}
