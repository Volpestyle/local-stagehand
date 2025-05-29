/**
 * Ollama Structured Output Client for Stagehand
 * Uses the official Ollama JavaScript client library
 *
 * Documentation:
 * - https://github.com/ollama/ollama-js
 * - https://ollama.com/blog/structured-outputs
 * - https://github.com/ollama/ollama/blob/main/docs/api.md
 */

import { Ollama, ChatRequest, ChatResponse, Options } from "ollama";
import { LogLine } from "../../types/log";
import { ClientOptions } from "../../types/model";
import { LLMCache } from "../cache/LLMCache";
import {
  LLMClient,
  CreateChatCompletionOptions,
  LLMResponse,
  ChatMessage,
} from "./LLMClient";
import { zodToJsonSchema } from "zod-to-json-schema";
import { CreateChatCompletionResponseError } from "@/types/stagehandErrors";
import { validateZodSchema } from "../utils";

export class OllamaClient extends LLMClient {
  public type = "ollama" as const;
  public hasVision = true; // Ollama supports vision models like llava, bakllava

  private client: Ollama;
  private logger: (logLine: LogLine) => void;
  private cache: LLMCache | undefined;
  private enableCaching: boolean;
  public clientOptions: ClientOptions;

  constructor({
    logger,
    enableCaching = false,
    cache,
    modelName,
    clientOptions,
  }: {
    logger: (message: LogLine) => void;
    enableCaching?: boolean;
    cache?: LLMCache;
    modelName: string;
    clientOptions?: ClientOptions;
  }) {
    super(modelName);
    this.logger = logger;
    this.cache = cache;
    this.enableCaching = enableCaching;
    this.clientOptions = clientOptions || {};

    // Initialize Ollama client
    this.client = new Ollama({
      host: clientOptions?.baseURL || "http://localhost:11434",
    });
  }

  /**
   * Convert ChatMessage format to Ollama's expected format
   */
  private formatMessage(message: ChatMessage): {
    role: "system" | "user" | "assistant";
    content: string;
    images?: string[];
  } {
    const formattedMessage: {
      role: "system" | "user" | "assistant";
      content: string;
      images?: string[];
    } = {
      role: message.role,
      content: "",
    };

    if (typeof message.content === "string") {
      formattedMessage.content = message.content;
    } else if (Array.isArray(message.content)) {
      const textParts: string[] = [];
      const imageParts: string[] = [];

      for (const part of message.content) {
        if ("text" in part && part.text) {
          textParts.push(part.text);
        } else if ("image_url" in part && part.image_url?.url) {
          // Extract base64 image data
          const base64Match = part.image_url.url.match(
            /^data:image\/[^;]+;base64,(.+)$/,
          );
          if (base64Match) {
            imageParts.push(base64Match[1]);
          }
        }
      }

      formattedMessage.content = textParts.join(" ");
      if (imageParts.length > 0) {
        formattedMessage.images = imageParts;
      }
    }

    return formattedMessage;
  }

  /**
   * Create chat completion with structured output support
   */
  async createChatCompletion<T = LLMResponse>(
    options: CreateChatCompletionOptions,
  ): Promise<T> {
    const {
      messages,
      temperature = 0.1,
      response_model,
      tools,
      requestId,
      top_p,
      frequency_penalty,
      maxTokens,
    } = options.options;

    // Check cache first
    const cacheOptions = {
      messages,
      temperature,
      response_model,
      tools,
      modelName: this.modelName,
    };

    if (this.enableCaching && requestId) {
      const cachedResponse = await this.cache?.get<T>(cacheOptions, requestId);
      if (cachedResponse) {
        this.logger({
          category: "llm_cache",
          message: "LLM cache hit - returning cached response",
          level: 1,
          auxiliary: {
            requestId: {
              value: requestId,
              type: "string",
            },
            modelName: {
              value: this.modelName,
              type: "string",
            },
          },
        });
        return cachedResponse as T;
      }
    }

    // Convert messages to Ollama format
    const formattedMessages = messages.map((msg) => this.formatMessage(msg));

    // Build Ollama options
    const ollamaOptions: Partial<Options> = {
      temperature,
      top_p,
      // Ollama uses repeat_penalty instead of frequency_penalty
      repeat_penalty: frequency_penalty ? 1 + frequency_penalty : undefined,
      num_predict: maxTokens,
    };

    // Build chat request
    const chatRequest: ChatRequest & { stream: false } = {
      model: this.modelName,
      messages: formattedMessages,
      stream: false, // Required for structured outputs and tools
      options: ollamaOptions,
      keep_alive: "5m", // Keep model loaded for 5 minutes
    };

    // Add structured output format if schema is provided
    if (response_model?.schema) {
      const jsonSchema = zodToJsonSchema(response_model.schema);
      chatRequest.format = jsonSchema;

      this.logger({
        category: "ollama",
        message: "Using structured output with schema",
        level: 1,
        auxiliary: {
          schemaKeys: {
            value: JSON.stringify(
              Object.keys(
                (jsonSchema as { properties?: Record<string, unknown> })
                  .properties || {},
              ),
            ),
            type: "object",
          },
        },
      });
    }

    // Add tools if provided
    if (tools && tools.length > 0) {
      chatRequest.tools = tools.map((tool) => ({
        type: "function" as const,
        function: {
          name: tool.name,
          description: tool.description || "",
          parameters: tool.parameters || {},
        },
      }));
    }

    let lastError: Error | unknown;
    const maxRetries = options.retries || 3;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        const response: ChatResponse = await this.client.chat(chatRequest);

        let parsedData = response.message.content;

        // Parse JSON if we have a response model
        if (response_model?.schema && typeof parsedData === "string") {
          try {
            parsedData = JSON.parse(parsedData);

            // Validate against schema
            const isValid = validateZodSchema(
              response_model.schema,
              parsedData,
            );
            if (!isValid) {
              throw new Error("Schema validation failed");
            }
          } catch (parseError) {
            if (attempt < maxRetries - 1) {
              this.logger({
                category: "ollama",
                message: `Failed to parse JSON response, retrying (attempt ${attempt + 1}/${maxRetries})`,
                level: 1,
                auxiliary: {
                  error: { value: String(parseError), type: "string" },
                  rawContent: {
                    value: parsedData.substring(0, 100),
                    type: "string",
                  },
                },
              });
              lastError = parseError;
              continue;
            }
            throw parseError;
          }
        }

        // Create LLM response in the expected format
        const llmResponse: LLMResponse = {
          id: `ollama-${Date.now()}`,
          object: "chat.completion",
          created: Math.floor(Date.now() / 1000),
          model: this.modelName,
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content:
                  typeof parsedData === "string"
                    ? parsedData
                    : JSON.stringify(parsedData),
                tool_calls: [],
              },
              finish_reason: "stop",
            },
          ],
          usage: {
            prompt_tokens: response.prompt_eval_count || 0,
            completion_tokens: response.eval_count || 0,
            total_tokens:
              (response.prompt_eval_count || 0) + (response.eval_count || 0),
          },
        };

        // Handle tool calls if present in the response
        if (
          response.message.tool_calls &&
          response.message.tool_calls.length > 0
        ) {
          llmResponse.choices[0].message.tool_calls =
            response.message.tool_calls.map((toolCall, index) => ({
              id: `call_${Date.now()}_${index}`,
              type: "function" as const,
              function: {
                name: toolCall.function.name,
                arguments: JSON.stringify(toolCall.function.arguments),
              },
            }));
        }

        // Transform to expected format
        const transformedResponse = response_model
          ? ({ data: parsedData, usage: llmResponse.usage } as T)
          : (llmResponse as T);

        // Cache the response
        if (this.enableCaching && requestId) {
          this.cache?.set(cacheOptions, transformedResponse, requestId);
        }

        return transformedResponse;
      } catch (error) {
        // Handle Ollama-specific errors
        if (error instanceof Error) {
          lastError = new CreateChatCompletionResponseError(
            `Ollama error: ${error.message}`,
          );
        } else {
          lastError = error;
        }

        if (attempt < maxRetries - 1) {
          this.logger({
            category: "ollama",
            message: `Request failed, retrying (attempt ${attempt + 1}/${maxRetries})`,
            level: 1,
            auxiliary: {
              error: { value: String(lastError), type: "string" },
            },
          });

          // Exponential backoff
          await new Promise((resolve) =>
            setTimeout(resolve, Math.pow(2, attempt) * 1000),
          );
        }
      }
    }

    this.logger({
      category: "ollama",
      message: "All retry attempts failed",
      level: 0,
      auxiliary: {
        error: { value: String(lastError), type: "string" },
        modelName: { value: this.modelName, type: "string" },
      },
    });

    throw lastError;
  }
}
