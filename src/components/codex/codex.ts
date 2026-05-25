import { css, html, LitElement, PropertyValues, unsafeCSS } from 'lit';
import { customElement, property, query, state } from 'lit/decorators.js';
import { ifDefined } from 'lit/directives/if-defined.js';
import type { Conversation } from '../../types/harmony-types';
import { parseCodexSession } from '../../utils/codex-session';
import type { EuphonyConversation } from '../conversation/conversation';
import type {
  FocusModeSettings,
  MessageLabelSettings
} from '../preference-window/preference-window';

import '../conversation/conversation';
import componentCSS from './codex.css?inline';

@customElement('euphony-codex')
export class EuphonyCodex extends LitElement {
  @property({ type: String, attribute: 'session-string' })
  sessionString = '';

  @property({ attribute: false })
  sessionData: unknown[] | null = null;

  @property({ type: String, attribute: 'sharing-url' })
  sharingURL: string | null = null;

  @property({ type: String, attribute: 'conversation-label' })
  conversationLabel = 'Session';

  @property({ type: String, attribute: 'conversation-max-width' })
  conversationMaxWidth: string | null = null;

  @property({ type: String, attribute: 'conversation-style' })
  conversationStyle = '';

  @property({ type: Boolean, attribute: 'preview-mode' })
  previewMode = false;

  @property({ type: Boolean, attribute: 'should-render-markdown' })
  shouldRenderMarkdown = false;

  @property({ type: Boolean, attribute: 'is-showing-metadata' })
  isShowingMetadata = false;

  @property({ type: Array, attribute: 'focus-mode-author' })
  focusModeAuthor: string[] = [];

  @property({ type: Array, attribute: 'focus-mode-recipient' })
  focusModeRecipient: string[] = [];

  @property({ type: Array, attribute: 'focus-mode-content-type' })
  focusModeContentType: string[] = [];

  @property({ type: Boolean, attribute: 'disable-markdown-button' })
  disableMarkdownButton = false;

  @property({ type: Boolean, attribute: 'disable-translation-button' })
  disableTranslationButton = false;

  @property({ type: Boolean, attribute: 'disable-share-button' })
  disableShareButton = false;

  @property({ type: Boolean, attribute: 'disable-metadata-button' })
  disableMetadataButton = false;

  @property({ type: Boolean, attribute: 'disable-message-metadata' })
  disableMessageMetadata = false;

  @property({ type: Boolean, attribute: 'disable-conversation-name' })
  disableConversationName = false;

  @property({ type: Boolean, attribute: 'disable-preference-button' })
  disablePreferenceButton = false;

  @property({ type: Boolean, attribute: 'disable-image-preview-window' })
  disableImagePreviewWindow = false;

  @property({ type: Boolean, attribute: 'disable-token-window' })
  disableTokenWindow = false;

  @property({ type: Boolean, attribute: 'disable-editing-mode-save-button' })
  disableEditingModeSaveButton = false;

  @property({ type: Boolean, attribute: 'disable-conversation-id-copy-button' })
  disableConversationIDCopyButton = false;

  @property({
    type: String,
    attribute: 'disable-download-convo-button-tooltip'
  })
  disableDownloadConvoButtonTooltip = '';

  @property({ type: String, attribute: 'disable-copy-convo-button-tooltip' })
  disableCopyConvoButtonTooltip = '';

  @property({ type: String, attribute: 'theme' })
  theme: 'auto' | 'light' | 'dark' = 'light';

  @state()
  conversation: Conversation | null = null;

  @state()
  parseError: string | null = null;

  @query('euphony-conversation')
  conversationComponent: EuphonyConversation | undefined;

  private parseSessionString(sessionString: string): unknown[] {
    const lines = sessionString
      .split('\n')
      .map(line => line.trim())
      .filter(line => line !== '');

    const events: unknown[] = [];
    for (const line of lines) {
      try {
        events.push(JSON.parse(line) as unknown);
      } catch (_error) {
        // Ignore malformed lines so one bad row does not drop the whole session.
      }
    }
    return events;
  }

  private refreshConversationFromSession() {
    const hasSessionData =
      Array.isArray(this.sessionData) && this.sessionData.length > 0;
    const rawEvents = hasSessionData
      ? (this.sessionData ?? [])
      : this.sessionString !== ''
        ? this.parseSessionString(this.sessionString)
        : [];

    const parseResult = parseCodexSession(rawEvents);
    if (!parseResult) {
      this.conversation = null;
      this.parseError =
        rawEvents.length === 0
          ? 'No Codex session data found.'
          : 'Unsupported or malformed Codex session JSONL.';
      return;
    }

    this.conversation = parseResult.conversation;
    this.parseError = null;
  }

  willUpdate(changedProperties: PropertyValues<this>) {
    if (
      changedProperties.has('sessionString') ||
      changedProperties.has('sessionData')
    ) {
      this.refreshConversationFromSession();
    }
  }

  render() {
    if (!this.conversation) {
      return html`
        <div class="empty-state">
          ${this.parseError ?? 'No Codex session to display.'}
        </div>
      `;
    }

    const focusModeAuthor =
      this.previewMode && this.focusModeAuthor.length === 0
        ? ['user', 'assistant']
        : this.focusModeAuthor;

    return html`
      <div class="codex-wrapper">
        <euphony-conversation
          .conversationData=${this.conversation}
          sharing-url=${ifDefined(this.sharingURL ?? undefined)}
          conversation-label=${this.conversationLabel}
          conversation-max-width=${ifDefined(
            this.conversationMaxWidth ?? undefined
          )}
          ?should-render-markdown=${this.shouldRenderMarkdown}
          ?is-showing-metadata=${this.isShowingMetadata}
          .focusModeAuthor=${focusModeAuthor}
          .focusModeRecipient=${this.focusModeRecipient}
          .focusModeContentType=${this.focusModeContentType}
          ?disable-markdown-button=${this.disableMarkdownButton}
          ?disable-translation-button=${this.disableTranslationButton}
          ?disable-share-button=${this.disableShareButton}
          ?disable-metadata-button=${this.disableMetadataButton}
          ?disable-message-metadata=${this.disableMessageMetadata}
          ?disable-conversation-name=${this.disableConversationName}
          ?disable-preference-button=${this.disablePreferenceButton}
          ?disable-image-preview-window=${this.disableImagePreviewWindow}
          ?disable-token-window=${this.disableTokenWindow}
          ?disable-editing-mode-save-button=${this.disableEditingModeSaveButton}
          ?disable-conversation-id-copy-button=${this
            .disableConversationIDCopyButton}
          disable-download-convo-button-tooltip=${ifDefined(
            this.disableDownloadConvoButtonTooltip || undefined
          )}
          disable-copy-convo-button-tooltip=${ifDefined(
            this.disableCopyConvoButtonTooltip || undefined
          )}
          theme=${this.theme}
          style=${this.conversationStyle}
        ></euphony-conversation>
      </div>
    `;
  }

  static styles = [
    css`
      ${unsafeCSS(componentCSS)}
    `
  ];

  preferenceWindowMessageLabelChanged(e: CustomEvent<MessageLabelSettings>) {
    this.conversationComponent?.preferenceWindowMessageLabelChanged(e);
  }

  preferenceWindowFocusModeSettingsChanged(e: CustomEvent<FocusModeSettings>) {
    this.conversationComponent?.preferenceWindowFocusModeSettingsChanged(e);
  }

  expandBlockContents() {
    this.conversationComponent?.expandBlockContents();
  }

  collapseBlockContents() {
    this.conversationComponent?.collapseBlockContents();
  }

  translationButtonClicked() {
    void this.conversationComponent?.translationButtonClicked();
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'euphony-codex': EuphonyCodex;
  }
}
