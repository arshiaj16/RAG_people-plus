from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session

import main 

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="CHANGE_THIS_TO_A_SECURE_RANDOM_KEY")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def root():
    return RedirectResponse("/login")


# ─── Signup ─────────────────────────────────────────────────────────────────────

@app.get("/create_user")
async def show_create_user(request: Request):
    return templates.TemplateResponse("create_user.html", {"request": request, "error": None})


@app.post("/create_user")
async def create_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    db: Session = main.SessionLocal()
    existing = db.query(main.User).filter(main.User.username == username).first()
    if existing:
        db.close()
        return templates.TemplateResponse(
            "create_user.html",
            {"request": request, "error": "Username already exists."}
        )

    # Persist new user
    user = main.User(username=username, password_hash=password)
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()

    # Auto-login
    request.session["username"] = username
    request.session["password"] = password
    return RedirectResponse("/chat", status_code=303)


# ─── Login ──────────────────────────────────────────────────────────────────────

@app.get("/login")
async def login_form(request: Request):
    if request.session.get("username"):
        return RedirectResponse("/chat")
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    db: Session = main.SessionLocal()
    user = db.query(main.User).filter_by(username=username).first()
    if user:
        if user.password_hash != password:
            db.close()
            return templates.TemplateResponse("login.html", {
                "request": request,
                "error": "Invalid credentials."
            })
    else:
        # Auto-register new user
        user = main.User(username=username, password_hash=password)
        db.add(user)
        db.commit()
        db.refresh(user)
    db.close()

    request.session["username"] = username
    request.session["password"] = password
    return RedirectResponse("/chat", status_code=303)


# ─── Chat Interface ─────────────────────────────────────────────────────────────

@app.get("/chat")
async def chat_page(request: Request):
    username = request.session.get("username")
    password = request.session.get("password")
    if not username or not password:
        return RedirectResponse("/login")

    # Fetch last 10 entries
    db: Session = main.SessionLocal()
    user = db.query(main.User).filter_by(username=username).first()
    db.close()
    if not user:
        return RedirectResponse("/login")

    history = main.get_user_history(user.user_id, limit=10)
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "username": username,
        "history": history,
        "status": None
    })


@app.post("/chat")
async def chat_submit(request: Request, query: str = Form(...)):
    username = request.session.get("username")
    password = request.session.get("password")
    if not username or not password:
        return RedirectResponse("/login")

    # Show thinking indicator by re-rendering with a status
    # (Client-side JS handles immediate "Thinking..." on form submit)

    # Invoke RAG pipeline (also stores Q/A)
    answer = main.rag_pipeline_with_history(username, password, query)

    # Reload history
    db: Session = main.SessionLocal()
    user = db.query(main.User).filter_by(username=username).first()
    db.close()
    history = main.get_user_history(user.user_id, limit=10)

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "username": username,
        "history": history,
        "status": None
    })

# ─── Logout ──────────────────────────────────────────────────────────────────────

@app.get("/logout")
async def logout(request: Request):
    # Clear the session
    request.session.pop("username", None)
    request.session.pop("password", None)
    # Redirect to the login page
    return RedirectResponse("/login", status_code=302)