package com.hyperbolic.geometry;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;

public class MainActivity extends Activity {
    private static final String URL = "https://hyperbolic-geometry-app.onrender.com";
    private WebView webView;

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        RelativeLayout layout = new RelativeLayout(this);
        layout.setBackgroundColor(0xFF1a1a2e);

        ProgressBar pb = new ProgressBar(this);
        RelativeLayout.LayoutParams pbParams = new RelativeLayout.LayoutParams(
            RelativeLayout.LayoutParams.WRAP_CONTENT,
            RelativeLayout.LayoutParams.WRAP_CONTENT
        );
        pbParams.addRule(RelativeLayout.CENTER_IN_PARENT);
        layout.addView(pb, pbParams);

        webView = new WebView(this);
        layout.addView(webView, new RelativeLayout.LayoutParams(
            RelativeLayout.LayoutParams.MATCH_PARENT,
            RelativeLayout.LayoutParams.MATCH_PARENT
        ));

        setContentView(layout);

        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setUseWideViewPort(true);
        settings.setLoadWithOverviewMode(true);

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageFinished(WebView v, String url) {
                pb.setVisibility(View.GONE);
                webView.setVisibility(View.VISIBLE);
            }
        });

        pb.setVisibility(View.VISIBLE);
        webView.setVisibility(View.INVISIBLE);
        webView.loadUrl(URL);
    }

    @Override
    public void onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }
}
