<div id="toc" align="center">
  <ul style="list-style: none">
    <a href="https://stagehand.dev">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://stagehand.dev/logo-dark.svg" />
        <img alt="Stagehand" src="https://stagehand.dev/logo-light.svg" />
      </picture>
    </a>
  </ul>
</div>

<p align="center">
  The production-ready framework for AI browser automations.<br>
  <a href="https://docs.stagehand.dev">Read the Docs</a>
</p>

<p align="center">
  <a href="https://github.com/browserbase/stagehand/tree/main?tab=MIT-1-ov-file#MIT-1-ov-file">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://stagehand.dev/api/assets/license?mode=dark" />
      <img alt="MIT License" src="https://stagehand.dev/api/assets/license?mode=light" />
    </picture>
  </a>
  <a href="https://stagehand.dev/slack">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://stagehand.dev/api/assets/slack?mode=dark" />
      <img alt="Slack Community" src="https://stagehand.dev/api/assets/slack?mode=light" />
    </picture>
  </a>
</p>

<p align="center">
	<a href="https://trendshift.io/repositories/12122" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12122" alt="browserbase%2Fstagehand | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

## 🦙 Ollama Integration

This fork adds complete **Ollama support** to Stagehand, enabling local LLM automation with structured outputs.

### Key Features

- **Official Ollama Client**: Built on the official [`ollama`](https://www.npmjs.com/package/ollama) npm package for maximum compatibility
- **Structured Outputs**: Full JSON schema support using Zod types for reliable data extraction
- **Vision Models**: Support for multi-modal models like LLaVA and BakLLaVA

### Quick Start

```typescript
const stagehand = new Stagehand({
  env: "LOCAL",
  modelName: "ollama/llama3.2:latest", // Use ollama/ prefix
  modelClientOptions: {
    baseURL: "http://localhost:11434", // Your Ollama server
  },
});
```

### Why the Official Package?

We use the official [`ollama`](https://www.npmjs.com/package/ollama) package instead of custom HTTP clients:

- **Type Safety**: Leverages official TypeScript types (`ChatRequest`, `ChatResponse`, `Options`)
- **API Compatibility**: Automatically stays compatible with Ollama API changes
- **Reliability**: Battle-tested HTTP handling, connection management, and error handling
- **Future-Proof**: New Ollama features (like tool calling) work immediately
- **Less Maintenance**: No need to maintain custom API client code

The `OllamaClient` acts as a bridge between Stagehand's `LLMClient` interface and the official Ollama package, handling message format conversion, schema validation, and response transformation automatically.

### Recommended Model: Qwen2.5-7B-Instruct-1M

This integration is optimized for **Qwen2.5-7B-Instruct-1M**, which provides exceptional performance for browser automation tasks:

```bash
# Install the recommended model
ollama pull yasserrmd/Qwen2.5-7B-Instruct-1M:latest
```

**Why Qwen2.5-7B-Instruct-1M?**

- **Ultra-Long Context**: 1,010,000 token context window handles complex web pages and extensive DOM trees without truncation
- **Structured Output Excellence**: Superior JSON schema adherence for reliable data extraction from web content
- **Instruction Following**: Fine-tuned specifically for instruction-following tasks, ideal for Stagehand's `observe()`, `act()`, and `extract()` operations
- **Efficient Architecture**: 7.61B parameters with sparse attention for optimal speed-to-quality ratio
- **Web Content Understanding**: Excellent performance on HTML/DOM analysis and element identification

The extended context length is particularly valuable for processing large accessibility trees from modern web applications, while the model's structured output capabilities ensure reliable job data extraction and form interaction.

## Why Stagehand?

Most existing browser automation tools either require you to write low-level code in a framework like Selenium, Playwright, or Puppeteer, or use high-level agents that can be unpredictable in production. By letting developers choose what to write in code vs. natural language, Stagehand is the natural choice for browser automations in production.

1. **Choose when to write code vs. natural language**: use AI when you want to navigate unfamiliar pages, and use code ([Playwright](https://playwright.dev/)) when you know exactly what you want to do.

2. **Preview and cache actions**: Stagehand lets you preview AI actions before running them, and also helps you easily cache repeatable actions to save time and tokens.

3. **Computer use models with one line of code**: Stagehand lets you integrate SOTA computer use models from OpenAI and Anthropic into the browser with one line of code.

## Example

Here's how to build a sample browser automation with Stagehand:

<div align="center">
  <div style="max-width:300px;">
    <img src="/media/github_demo.gif" alt="See Stagehand in Action">
  </div>
</div>

```typescript
// Use Playwright functions on the page object
const page = stagehand.page;
await page.goto("https://github.com/browserbase");

// Use act() to execute individual actions
await page.act("click on the stagehand repo");

// Use Computer Use agents for larger actions
const agent = stagehand.agent({
  provider: "openai",
  model: "computer-use-preview",
});
await agent.execute("Get to the latest PR");

// Use extract() to read data from the page
const { author, title } = await page.extract({
  instruction: "extract the author and title of the PR",
  schema: z.object({
    author: z.string().describe("The username of the PR author"),
    title: z.string().describe("The title of the PR"),
  }),
});
```

## Documentation

Visit [docs.stagehand.dev](https://docs.stagehand.dev) to view the full documentation.

## Getting Started

Start with Stagehand with one line of code, or check out our [Quickstart Guide](https://docs.stagehand.dev/get_started/quickstart) for more information:

```bash
npx create-browser-app
```

<div align="center">
    <a href="https://www.loom.com/share/f5107f86d8c94fa0a8b4b1e89740f7a7">
      <p>Watch Anirudh demo create-browser-app to create a Stagehand project!</p>
    </a>
    <a href="https://www.loom.com/share/f5107f86d8c94fa0a8b4b1e89740f7a7">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/f5107f86d8c94fa0a8b4b1e89740f7a7-ec3f428b6775ceeb-full-play.gif">
    </a>
  </div>

### Build and Run from Source

```bash
git clone https://github.com/browserbase/stagehand.git
cd stagehand
pnpm install
pnpm playwright install
pnpm run build
pnpm run example # run the blank script at ./examples/example.ts
pnpm run example 2048 # run the 2048 example at ./examples/2048.ts
```

Stagehand is best when you have an API key for an LLM provider and Browserbase credentials. To add these to your project, run:

```bash
cp .env.example .env
nano .env # Edit the .env file to add API keys
```

## Contributing

> [!NOTE]  
> We highly value contributions to Stagehand! For questions or support, please join our [Slack community](https://stagehand.dev/slack).

At a high level, we're focused on improving reliability, speed, and cost in that order of priority. If you're interested in contributing, we strongly recommend reaching out to [Anirudh Kamath](https://x.com/kamathematic) or [Paul Klein](https://x.com/pk_iv) in our [Slack community](https://stagehand.dev/slack) before starting to ensure that your contribution aligns with our goals.

For more information, please see our [Contributing Guide](https://docs.stagehand.dev/examples/contributing).

## Acknowledgements

This project heavily relies on [Playwright](https://playwright.dev/) as a resilient backbone to automate the web. It also would not be possible without the awesome techniques and discoveries made by [tarsier](https://github.com/reworkd/tarsier), [gemini-zod](https://github.com/jbeoris/gemini-zod), and [fuji-web](https://github.com/normal-computing/fuji-web).

We'd like to thank the following people for their major contributions to Stagehand:

- [Paul Klein](https://github.com/pkiv)
- [Anirudh Kamath](https://github.com/kamath)
- [Sean McGuire](https://github.com/seanmcguire12)
- [Miguel Gonzalez](https://github.com/miguelg719)
- [Sameel Arif](https://github.com/sameelarif)
- [Filip Michalsky](https://github.com/filip-michalsky)
- [Jeremy Press](https://x.com/jeremypress)
- [Navid Pour](https://github.com/navidpour)

## License

Licensed under the MIT License.

Copyright 2025 Browserbase, Inc.
