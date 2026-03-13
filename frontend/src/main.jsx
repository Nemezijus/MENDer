import React from 'react';
import { createRoot } from 'react-dom/client';
import {
  MantineProvider,
  createTheme,
  Card,
  Tabs,
  Input,
  InputWrapper,
  Select,
  TextInput,
  Textarea,
  NumberInput,
  Button,
  Checkbox,
} from '@mantine/core';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import '@mantine/core/styles.css';
import './index.css';
import App from './app/App.jsx';

const queryClient = new QueryClient();

const theme = createTheme({
  components: {
    Card: Card.extend({
      styles: {
        root: {
          backgroundColor: 'var(--app-card-bg, var(--app-color-surface))',
          color: 'var(--app-card-fg, var(--app-color-text))',
        },
      },
      vars: () => ({
        root: {
          '--paper-border-color': 'var(--app-card-border, var(--app-color-c4))',
        },
      }),
    }),

    Tabs: Tabs.extend({}),

    Input: Input.extend({
      classNames: {
        input: 'appFieldInput',
        section: 'appFieldSection',
      },
    }),

    InputWrapper: InputWrapper.extend({
      classNames: {
        label: 'appFieldLabel',
        description: 'appFieldDescription',
        error: 'appFieldError',
      },
    }),

    Select: Select.extend({
      classNames: {
        input: 'appFieldInput',
        section: 'appFieldSection',
        label: 'appFieldLabel',
        description: 'appFieldDescription',
        error: 'appFieldError',
        dropdown: 'appFieldDropdown',
        options: 'appFieldOptions',
        option: 'appFieldOption',
        empty: 'appFieldEmpty',
        groupLabel: 'appFieldGroupLabel',
      },
    }),

    TextInput: TextInput.extend({}),
    Textarea: Textarea.extend({}),
    NumberInput: NumberInput.extend({}),

    Button: Button.extend({
      classNames: {
        root: 'appButtonRoot',
        label: 'appButtonLabel',
        section: 'appButtonSection',
      },
    }),

    Checkbox: Checkbox.extend({
      classNames: {
        root: 'appCheckboxRoot',
        body: 'appCheckboxBody',
        inner: 'appCheckboxInner',
        input: 'appCheckboxInput',
        icon: 'appCheckboxIcon',
        label: 'appCheckboxLabel',
        description: 'appCheckboxDescription',
        error: 'appCheckboxError',
      },
    }),
  },
  defaultRadius: 0,
});

const rootEl = document.getElementById('root');

createRoot(rootEl).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <MantineProvider theme={theme} defaultColorScheme="light">
        <App />
      </MantineProvider>
    </QueryClientProvider>
  </React.StrictMode>
);