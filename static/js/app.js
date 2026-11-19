const { createApp, ref, reactive, computed, onMounted, nextTick } = Vue;

createApp({
  setup() {
    const currentView = ref("home");
    const user = ref(null);
    const loading = ref(false);
    const dashboardLoading = ref(false);
    const comparisonLoading = ref(false);
    const artifactsLoading = ref(false);
    const message = reactive({ text: "", type: "" });

    const loginForm = reactive({ username: "", password: "" });
    const registerForm = reactive({ username: "", password: "", confirm_password: "" });
    const predictForm = reactive({ customer_unique_id: "" });

    const predictResult = ref(null);
    const batchResults = ref([]);
    const batchFilename = ref("");
    const modelComparison = ref([]);
    const featureImportance = ref([]);
    const chartInstance = ref(null);

    const artifacts = reactive({
      comparison: null,
      metric_files: [],
      prediction_files: [],
      image_groups: []
    });

    const dashboard = reactive({
      summary: {
        record_count: 0,
        user_count: 0,
        best_model: "-",
        best_f1: 0,
        top_categories: {},
        history_count: 0
      },
      history: [],
      result_paths: {
        comparison_csv: "",
        database: "",
        processed_data: "",
        models_dir: ""
      }
    });

    const isAuthenticated = computed(() => !!user.value);
    const topCategories = computed(() => Object.entries(dashboard.summary.top_categories || {}));

    function setMessage(text, type = "success") {
      message.text = text;
      message.type = type;
    }

    function clearMessage() {
      message.text = "";
      message.type = "";
    }

    async function request(url, options = {}) {
      const response = await fetch(url, {
        credentials: "include",
        ...options
      });

      let data;
      try {
        data = await response.json();
      } catch (_error) {
        throw new Error("后端服务未返回有效数据，请确认系统服务已经正常启动。");
      }

      if (!response.ok || data.success === false) {
        throw new Error(data.message || "请求失败，请稍后重试。");
      }

      return data;
    }

    async function loadSession() {
      try {
        const data = await request("/api/session");
        user.value = data.user;
        if (data.authenticated) {
          currentView.value = "dashboard";
          await Promise.all([loadDashboard(), loadModelComparison(), loadArtifacts(), loadFeatureImportance()]);
        }
      } catch (error) {
        user.value = null;
        setMessage(error.message, "error");
      }
    }

    async function loadDashboard() {
      dashboardLoading.value = true;
      try {
        const data = await request("/api/dashboard");
        dashboard.summary = data.summary;
        dashboard.history = data.history;
        dashboard.result_paths = data.result_paths;
      } catch (error) {
        setMessage(error.message, "error");
      } finally {
        dashboardLoading.value = false;
      }
    }

    async function loadModelComparison() {
      comparisonLoading.value = true;
      try {
        const data = await request("/api/model-comparison");
        modelComparison.value = data.models || [];
        await nextTick();
        renderComparisonChart();
      } catch (error) {
        setMessage(error.message, "error");
      } finally {
        comparisonLoading.value = false;
      }
    }

    async function loadArtifacts() {
      artifactsLoading.value = true;
      try {
        const data = await request("/api/artifacts");
        artifacts.comparison = data.artifacts.comparison;
        artifacts.metric_files = data.artifacts.metric_files || [];
        artifacts.prediction_files = data.artifacts.prediction_files || [];
        artifacts.image_groups = data.artifacts.image_groups || [];
      } catch (error) {
        setMessage(error.message, "error");
      } finally {
        artifactsLoading.value = false;
      }
    }

    async function loadFeatureImportance() {
      try {
        const data = await request("/api/feature-importance");
        featureImportance.value = data.features || [];
      } catch (error) {
        setMessage(error.message, "error");
      }
    }

    function renderComparisonChart() {
      const canvas = document.getElementById("modelComparisonChart");
      if (!canvas || modelComparison.value.length === 0) {
        return;
      }

      if (chartInstance.value) {
        chartInstance.value.destroy();
      }

      const labels = modelComparison.value.map(item => item.model);
      chartInstance.value = new Chart(canvas, {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: "Accuracy",
              data: modelComparison.value.map(item => Number(item.accuracy || 0)),
              borderColor: "#b55233",
              backgroundColor: "rgba(181, 82, 51, 0.12)",
              tension: 0.35
            },
            {
              label: "Precision",
              data: modelComparison.value.map(item => Number(item.precision || 0)),
              borderColor: "#1f6d63",
              backgroundColor: "rgba(31, 109, 99, 0.12)",
              tension: 0.35
            },
            {
              label: "Recall",
              data: modelComparison.value.map(item => Number(item.recall || 0)),
              borderColor: "#4a5acb",
              backgroundColor: "rgba(74, 90, 203, 0.12)",
              tension: 0.35
            },
            {
              label: "F1",
              data: modelComparison.value.map(item => Number(item.f1 || 0)),
              borderColor: "#9b6c12",
              backgroundColor: "rgba(155, 108, 18, 0.12)",
              tension: 0.35
            },
            {
              label: "ROC AUC",
              data: modelComparison.value.map(item => Number(item.roc_auc || 0)),
              borderColor: "#7f3c8d",
              backgroundColor: "rgba(127, 60, 141, 0.12)",
              tension: 0.35
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "bottom"
            }
          },
          scales: {
            y: {
              min: 0,
              max: 1
            }
          }
        }
      });
    }

    async function register() {
      loading.value = true;
      clearMessage();
      try {
        const data = await request("/api/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(registerForm)
        });
        setMessage(data.message, "success");
        registerForm.username = "";
        registerForm.password = "";
        registerForm.confirm_password = "";
        currentView.value = "login";
      } catch (error) {
        setMessage(error.message, "error");
      } finally {
        loading.value = false;
      }
    }

    async function login() {
      loading.value = true;
      clearMessage();
      try {
        const data = await request("/api/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(loginForm)
        });
        user.value = data.user;
        loginForm.password = "";
        currentView.value = "dashboard";
        setMessage(data.message, "success");
        await Promise.all([loadDashboard(), loadModelComparison(), loadArtifacts(), loadFeatureImportance()]);
      } catch (error) {
        setMessage(error.message, "error");
      } finally {
        loading.value = false;
      }
    }

    async function logout() {
      try {
        const data = await request("/api/logout", { method: "POST" });
        user.value = null;
        predictResult.value = null;
        batchResults.value = [];
        batchFilename.value = "";
        modelComparison.value = [];
        featureImportance.value = [];
        artifacts.comparison = null;
        artifacts.metric_files = [];
        artifacts.prediction_files = [];
        artifacts.image_groups = [];
        if (chartInstance.value) {
          chartInstance.value.destroy();
          chartInstance.value = null;
        }
        currentView.value = "home";
        setMessage(data.message, "success");
      } catch (error) {
        setMessage(error.message, "error");
      }
    }

    async function predictSingle() {
      loading.value = true;
      clearMessage();
      try {
        const data = await request("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(predictForm)
        });
        predictResult.value = data.result;
        setMessage(data.message, "success");
        await loadDashboard();
      } catch (error) {
        setMessage(error.message, "error");
      } finally {
        loading.value = false;
      }
    }

    async function predictBatch(event) {
      const file = event.target.files[0];
      if (!file) {
        return;
      }

      const formData = new FormData();
      formData.append("data_file", file);

      loading.value = true;
      clearMessage();
      try {
        const data = await request("/api/batch-predict", {
          method: "POST",
          body: formData
        });
        batchFilename.value = data.filename;
        batchResults.value = data.results || [];
        setMessage(data.message, "success");
      } catch (error) {
        setMessage(error.message, "error");
      } finally {
        loading.value = false;
        event.target.value = "";
      }
    }

    function go(view) {
      clearMessage();
      currentView.value = view;
      if (view === "dashboard" && isAuthenticated.value) {
        loadDashboard();
        loadModelComparison();
        loadArtifacts();
        loadFeatureImportance();
      }
    }

    onMounted(loadSession);

    return {
      currentView,
      user,
      loading,
      dashboardLoading,
      comparisonLoading,
      artifactsLoading,
      message,
      loginForm,
      registerForm,
      predictForm,
      predictResult,
      batchResults,
      batchFilename,
      dashboard,
      modelComparison,
      featureImportance,
      artifacts,
      isAuthenticated,
      topCategories,
      go,
      register,
      login,
      logout,
      predictSingle,
      predictBatch
    };
  },
  template: `
    <div class="page-shell">
      <header class="site-header">
        <div>
          <p class="eyebrow">Vue Frontend + Flask API</p>
          <h1 class="site-title">电商用户购买行为预测系统</h1>
        </div>
        <nav class="site-nav">
          <template v-if="isAuthenticated">
            <a href="#" @click.prevent="go('dashboard')">系统总览</a>
            <a href="#" @click.prevent="go('predict')">单用户预测</a>
            <a href="#" @click.prevent="go('batch')">批量预测</a>
            <a href="#" @click.prevent="logout">退出登录</a>
          </template>
          <template v-else>
            <a href="#" @click.prevent="go('home')">首页</a>
            <a href="#" @click.prevent="go('login')">登录</a>
            <a href="#" @click.prevent="go('register')">注册</a>
          </template>
        </nav>
      </header>

      <section v-if="message.text" class="flash-stack">
        <div class="flash" :class="'flash-' + message.type">{{ message.text }}</div>
      </section>

      <main class="content">
        <section v-if="currentView === 'home'" class="hero">
          <div class="hero-copy">
            <p class="eyebrow">Graduation Project Demo</p>
            <h2>基于深度学习的电商用户购买行为预测与可视化分析平台</h2>
            <p class="hero-text">
              本系统围绕电商用户未来购买行为预测任务展开设计，集成了用户身份认证、
              单用户预测、批量预测、模型效果对比和实验结果展示等功能。
            </p>
            <div class="hero-actions">
              <a class="button button-primary" href="#" @click.prevent="go('register')">立即注册</a>
              <a class="button button-secondary" href="#" @click.prevent="go('login')">已有账号，进入系统</a>
            </div>
          </div>

          <div class="hero-card">
            <div class="metric-card">
              <span>前端框架</span>
              <strong>Vue 3 + Chart.js</strong>
            </div>
            <div class="metric-card">
              <span>后端架构</span>
              <strong>Flask + SQLite</strong>
            </div>
            <div class="metric-card">
              <span>展示内容</span>
              <strong>预测结果、模型对比、ROC 曲线、混淆矩阵</strong>
            </div>
          </div>
        </section>

        <section v-if="currentView === 'login'" class="auth-layout">
          <form class="panel form-panel" @submit.prevent="login">
            <h2>用户登录</h2>
            <label>用户名</label>
            <input v-model="loginForm.username" type="text" placeholder="请输入用户名" required>
            <label>密码</label>
            <input v-model="loginForm.password" type="password" placeholder="请输入登录密码" required>
            <button class="button button-primary" type="submit" :disabled="loading">登录系统</button>
            <p class="muted-text">若尚未注册账号，请先 <a href="#" @click.prevent="go('register')">完成注册</a></p>
          </form>
        </section>

        <section v-if="currentView === 'register'" class="auth-layout">
          <form class="panel form-panel" @submit.prevent="register">
            <h2>用户注册</h2>
            <label>用户名</label>
            <input v-model="registerForm.username" type="text" placeholder="请输入用户名" required>
            <label>密码</label>
            <input v-model="registerForm.password" type="password" placeholder="请输入不少于 6 位的密码" required>
            <label>确认密码</label>
            <input v-model="registerForm.confirm_password" type="password" placeholder="请再次输入密码" required>
            <button class="button button-primary" type="submit" :disabled="loading">完成注册</button>
            <p class="muted-text">如果已经拥有账号，可直接 <a href="#" @click.prevent="go('login')">返回登录</a></p>
          </form>
        </section>

        <template v-if="currentView === 'dashboard' && isAuthenticated">
          <section class="dashboard-top">
            <div class="panel welcome-panel">
              <p class="eyebrow">Dashboard</p>
              <h2>{{ user.username }}，欢迎进入系统总览页面</h2>
              <p>该页面集中展示数据规模、模型表现、实验图像与预测记录。</p>
            </div>
          </section>

          <section v-if="dashboardLoading" class="panel">
            <p class="muted-text">系统概览数据加载中，请稍候...</p>
          </section>

          <section v-else class="metrics-grid">
            <div class="panel metric-box">
              <span>数据记录总数</span>
              <strong>{{ dashboard.summary.record_count }}</strong>
            </div>
            <div class="panel metric-box">
              <span>用户样本数量</span>
              <strong>{{ dashboard.summary.user_count }}</strong>
            </div>
            <div class="panel metric-box">
              <span>当前最佳模型</span>
              <strong>{{ dashboard.summary.best_model }}</strong>
            </div>
            <div class="panel metric-box">
              <span>最佳 F1 分数</span>
              <strong>{{ Number(dashboard.summary.best_f1 || 0).toFixed(4) }}</strong>
            </div>
          </section>

          <section class="panel">
            <h3>模型性能对比曲线</h3>
            <p class="muted-text">该图基于模型测试结果绘制，用于展示不同模型在 Accuracy、Precision、Recall、F1 和 ROC AUC 等指标上的综合表现。</p>
            <p v-if="comparisonLoading" class="muted-text">模型性能对比图加载中，请稍候...</p>
            <div v-if="modelComparison.length" class="chart-wrap">
              <canvas id="modelComparisonChart"></canvas>
            </div>
            <p v-else-if="!comparisonLoading" class="muted-text">当前暂无可展示的模型对比结果。</p>
          </section>

          <section v-if="modelComparison.length" class="panel">
            <h3>模型指标对比表</h3>
            <table class="data-table">
              <thead>
                <tr>
                  <th>模型名称</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>ROC AUC</th>
                  <th>Category Top-3</th>
                  <th>Category Macro-F1</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in modelComparison" :key="row.model">
                  <td>{{ row.model }}</td>
                  <td>{{ Number(row.accuracy || 0).toFixed(4) }}</td>
                  <td>{{ Number(row.precision || 0).toFixed(4) }}</td>
                  <td>{{ Number(row.recall || 0).toFixed(4) }}</td>
                  <td>{{ Number(row.f1 || 0).toFixed(4) }}</td>
                  <td>{{ Number(row.roc_auc || 0).toFixed(4) }}</td>
                  <td>{{ Number(row.category_top3_accuracy || 0).toFixed(4) }}</td>
                  <td>{{ Number(row.category_macro_f1 || 0).toFixed(4) }}</td>
                </tr>
              </tbody>
            </table>
          </section>

          <section class="panel">
            <h3>实验结果图像展示</h3>
            <p class="muted-text">此处展示各模型训练曲线、ROC 曲线和混淆矩阵，可用于说明模型训练过程与分类效果。</p>
            <p v-if="artifactsLoading" class="muted-text">实验结果图像加载中，请稍候...</p>
            <div v-else class="artifact-grid">
              <article v-for="group in artifacts.image_groups" :key="group.model" class="artifact-card">
                <h4>{{ group.model.toUpperCase() }}</h4>
                <div class="artifact-preview">
                  <img v-if="group.training_history.exists" :src="group.training_history.url" :alt="group.model + ' 训练曲线'">
                  <p v-else class="muted-text">当前缺少该模型的训练曲线图片</p>
                </div>
                <div class="artifact-links">
                  <a v-if="group.training_history.exists" :href="group.training_history.url" target="_blank" rel="noreferrer">查看训练曲线</a>
                  <a v-if="group.roc_curve.exists" :href="group.roc_curve.url" target="_blank" rel="noreferrer">查看 ROC 曲线</a>
                  <a v-if="group.confusion_matrix.exists" :href="group.confusion_matrix.url" target="_blank" rel="noreferrer">查看混淆矩阵</a>
                  <a v-if="group.permutation_importance.exists" :href="group.permutation_importance.url" target="_blank" rel="noreferrer">查看特征重要性</a>
                </div>
              </article>
            </div>
          </section>

          <section v-if="featureImportance.length" class="panel">
            <h3>特征重要性分析</h3>
            <p class="muted-text">该表优先读取最佳模型的 permutation importance 结果；若训练图尚未生成，则使用相关系数代理分析作为兜底展示。</p>
            <table class="data-table">
              <thead>
                <tr>
                  <th>特征名称</th>
                  <th>特征类型</th>
                  <th>重要性分数</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in featureImportance" :key="row.feature">
                  <td>{{ row.feature }}</td>
                  <td>{{ row.feature_type || 'proxy' }}</td>
                  <td>{{ Number(row.importance || row.importance_score || 0).toFixed(4) }}</td>
                </tr>
              </tbody>
            </table>
          </section>

          <section class="two-column">
            <div class="panel">
              <h3>训练样本类别分布 Top 5</h3>
              <ul v-if="topCategories.length" class="simple-list">
                <li v-for="[category, count] in topCategories" :key="category">
                  {{ category }}
                  <span>{{ count }}</span>
                </li>
              </ul>
              <p v-else class="muted-text">当前暂无可展示的类别统计信息。</p>
            </div>

            <div class="panel">
              <h3>最近预测记录</h3>
              <table v-if="dashboard.history.length" class="data-table">
                <thead>
                  <tr>
                    <th>用户 ID</th>
                    <th>购买概率</th>
                    <th>推荐类别</th>
                    <th>预测时间</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="row in dashboard.history" :key="row.created_at + row.customer_unique_id">
                    <td>{{ row.customer_unique_id }}</td>
                    <td>{{ Number(row.purchase_probability).toFixed(4) }}</td>
                    <td>{{ row.recommended_category || '暂无推荐' }}</td>
                    <td>{{ row.created_at }}</td>
                  </tr>
                </tbody>
              </table>
              <p v-else class="muted-text">当前还没有生成预测记录。</p>
            </div>
          </section>

          <section class="panel">
            <h3>结果文件与保存位置</h3>
            <div class="path-list">
              <p class="muted-text">模型对比结果文件：{{ dashboard.result_paths.comparison_csv || '暂未找到' }}</p>
              <p class="muted-text">处理后数据文件：{{ dashboard.result_paths.processed_data || '暂未找到' }}</p>
              <p class="muted-text">系统数据库文件：{{ dashboard.result_paths.database || '暂未找到' }}</p>
              <p class="muted-text">模型文件目录：{{ dashboard.result_paths.models_dir || '暂未找到' }}</p>
            </div>
            <div class="artifact-links">
              <a v-if="artifacts.comparison && artifacts.comparison.exists" :href="artifacts.comparison.url" target="_blank" rel="noreferrer">打开模型对比 CSV</a>
              <template v-for="item in artifacts.metric_files" :key="item.filename">
                <a v-if="item.exists" :href="item.url" target="_blank" rel="noreferrer">{{ item.label }}</a>
              </template>
              <template v-for="item in artifacts.prediction_files" :key="item.filename">
                <a v-if="item.exists" :href="item.url" target="_blank" rel="noreferrer">{{ item.label }}</a>
              </template>
            </div>
          </section>
        </template>

        <section v-if="currentView === 'predict' && isAuthenticated" class="two-column">
          <form class="panel form-panel" @submit.prevent="predictSingle">
            <h2>单用户预测</h2>
            <p class="muted-text">请输入数据集中真实存在的 <code>customer_unique_id</code>，系统将输出该用户未来购买倾向及推荐类别。</p>
            <label>用户 ID</label>
            <input v-model="predictForm.customer_unique_id" type="text" placeholder="请输入待预测的用户 ID" required>
            <button class="button button-primary" type="submit" :disabled="loading">
              {{ loading ? '预测中...' : '开始预测' }}
            </button>
          </form>

          <div class="panel result-panel">
            <h2>预测结果展示</h2>
            <div v-if="predictResult" class="result-card">
              <p><strong>用户 ID：</strong>{{ predictResult.user_id }}</p>
              <p><strong>样本时间：</strong>{{ predictResult.snapshot_time }}</p>
              <p><strong>历史订单数：</strong>{{ predictResult.history_orders }}</p>
              <p><strong>预测购买概率：</strong>{{ Number(predictResult.purchase_probability).toFixed(4) }}</p>
              <p><strong>预测是否购买：</strong>{{ predictResult.predicted_purchase === 1 ? '是' : '否' }}</p>
              <p><strong>推荐商品类别：</strong>{{ predictResult.recommended_category || '暂无推荐' }}</p>
              <p><strong>采用模型：</strong>{{ predictResult.model_name }}</p>
            </div>
            <p v-else class="muted-text">提交用户 ID 后，系统将在此处展示对应的预测结果。</p>
          </div>
        </section>

        <section v-if="currentView === 'batch' && isAuthenticated" class="panel">
          <h2>批量预测</h2>
          <p class="muted-text">上传包含用户行为数据的 CSV 文件后，系统将自动完成批量推理，并展示前 50 条预测结果。</p>
          <label>选择 CSV 文件</label>
          <input type="file" accept=".csv" @change="predictBatch">
          <p v-if="batchFilename" class="muted-text">当前已上传文件：{{ batchFilename }}</p>
          <p v-if="loading" class="muted-text">批量预测执行中，请稍候...</p>

          <table v-if="batchResults.length" class="data-table">
            <thead>
              <tr>
                <th>用户 ID</th>
                <th>购买概率</th>
                <th>是否购买</th>
                <th>推荐类别</th>
                <th>使用模型</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in batchResults" :key="row.user_id + row.snapshot_time">
                <td>{{ row.user_id }}</td>
                <td>{{ Number(row.purchase_probability).toFixed(4) }}</td>
                <td>{{ row.predicted_purchase === 1 ? '是' : '否' }}</td>
                <td>{{ row.recommended_category || '暂无推荐' }}</td>
                <td>{{ row.model_name }}</td>
              </tr>
            </tbody>
          </table>
        </section>
      </main>
    </div>
  `
}).mount("#app");
