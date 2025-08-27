#!/usr/bin/env bash
#
# UZH Master Project - System Startup Script
# Complete end-to-end real-time transcription system with RAG evaluation
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT=50051
WEB_PORT=3000

print_header() {
    echo -e "${BLUE}"
    echo "================================================================================"
    echo "                      UZH MASTER PROJECT - STARTUP SCRIPT"
    echo "     Real-time Speech-to-Text Transcription with RAG Evaluation System"
    echo "================================================================================"
    echo -e "${NC}"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    print_status "Checking system prerequisites..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js v22+"
        exit 1
    fi
    
    # Check pnpm
    if ! command -v pnpm &> /dev/null; then
        print_error "pnpm is not installed. Please install pnpm v10.14+"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python is not installed. Please install Python 3.12+"
        exit 1
    fi
    
    # Check uv
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install uv v0.8+"
        exit 1
    fi
    
    # Check FFmpeg (required for transcription)
    if ! command -v ffmpeg &> /dev/null; then
        print_warning "FFmpeg not found. It's required for audio processing."
        print_warning "Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Ubuntu)"
    fi
    
    print_status "Prerequisites check completed ✓"
}

install_dependencies() {
    print_status "Installing dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # Install Node.js dependencies
    print_status "Installing Node.js dependencies with pnpm..."
    pnpm install
    
    # Generate protocol buffers
    print_status "Generating protocol buffers..."
    cd packages/protos
    pnpm run build
    
    cd "$PROJECT_ROOT"
    print_status "Dependencies installed ✓"
}

start_api_server() {
    print_status "Starting API server on port $API_PORT..."
    
    cd "$PROJECT_ROOT/packages/api"
    
    # Check if port is already in use
    if lsof -Pi :$API_PORT -sTCP:LISTEN -t >/dev/null; then
        print_warning "Port $API_PORT is already in use. Stopping existing process..."
        pkill -f "main.py" || true
        sleep 2
    fi
    
    # Start the API server in background
    print_status "Launching gRPC transcription server..."
    pnpm run start &
    API_PID=$!
    
    # Wait for server to start
    sleep 3
    
    # Check if server is running
    if ps -p $API_PID > /dev/null; then
        print_status "✓ API server started successfully (PID: $API_PID)"
        echo $API_PID > /tmp/uzh_api_server.pid
    else
        print_error "Failed to start API server"
        exit 1
    fi
}

start_web_frontend() {
    print_status "Starting web frontend on port $WEB_PORT..."
    
    cd "$PROJECT_ROOT/packages/web"
    
    # Check if port is already in use
    if lsof -Pi :$WEB_PORT -sTCP:LISTEN -t >/dev/null; then
        print_warning "Port $WEB_PORT is already in use. You may need to stop the existing process."
    fi
    
    # Start the web frontend in background
    print_status "Launching Next.js frontend..."
    pnpm run dev &
    WEB_PID=$!
    
    # Wait for frontend to start
    sleep 5
    
    # Check if frontend is running
    if ps -p $WEB_PID > /dev/null; then
        print_status "✓ Web frontend started successfully (PID: $WEB_PID)"
        echo $WEB_PID > /tmp/uzh_web_frontend.pid
    else
        print_error "Failed to start web frontend"
        exit 1
    fi
}

run_evaluation_demo() {
    print_status "Running RAG evaluation demo..."
    
    cd "$PROJECT_ROOT/packages/evaluation"
    
    # Configure Python environment
    print_status "Setting up Python environment for evaluation..."
    
    # Run the final pipeline demo
    print_status "Executing end-to-end pipeline evaluation..."
    uv run final_pipeline_demo.py
    
    print_status "✓ Evaluation demo completed"
}

show_startup_summary() {
    echo -e "${GREEN}"
    echo "================================================================================"
    echo "                            SYSTEM STARTUP COMPLETE"
    echo "================================================================================"
    echo -e "${NC}"
    echo ""
    echo -e "${BLUE}🚀 SERVICES RUNNING:${NC}"
    echo "   • gRPC API Server:  http://localhost:$API_PORT"
    echo "   • Web Frontend:     http://localhost:$WEB_PORT"
    echo ""
    echo -e "${BLUE}📋 WHAT YOU CAN DO:${NC}"
    echo "   1. Open http://localhost:$WEB_PORT in your browser"
    echo "   2. Click 'Start Recording' to begin real-time transcription"
    echo "   3. Speak into your microphone"
    echo "   4. View transcriptions and RAG search results in real-time"
    echo ""
    echo -e "${BLUE}🔧 SYSTEM COMPONENTS:${NC}"
    echo "   • Real-time audio capture and streaming"
    echo "   • gRPC-based transcription service (Faster Whisper)"
    echo "   • RAG (Retrieval-Augmented Generation) service"
    echo "   • Persistent recording storage"
    echo "   • MS MARCO evaluation framework"
    echo ""
    echo -e "${BLUE}📊 EVALUATION SYSTEM:${NC}"
    echo "   • Recording → Transcription → RAG → MS MARCO Evaluation"
    echo "   • Performance metrics: Precision@k, Recall@k, NDCG@k, MRR"
    echo "   • Comparative analysis against standard benchmarks"
    echo ""
    echo -e "${YELLOW}⚠️  SYSTEM MANAGEMENT:${NC}"
    echo "   • Stop all services: ./stop-system.sh"
    echo "   • View logs: tail -f /tmp/uzh_*.log"
    echo "   • API Server PID: $(cat /tmp/uzh_api_server.pid 2>/dev/null || echo 'N/A')"
    echo "   • Web Frontend PID: $(cat /tmp/uzh_web_frontend.pid 2>/dev/null || echo 'N/A')"
    echo ""
    echo -e "${GREEN}Press Ctrl+C to stop all services${NC}"
}

cleanup() {
    print_status "Shutting down services..."
    
    # Kill API server
    if [ -f /tmp/uzh_api_server.pid ]; then
        API_PID=$(cat /tmp/uzh_api_server.pid)
        if ps -p $API_PID > /dev/null; then
            kill $API_PID
            print_status "Stopped API server (PID: $API_PID)"
        fi
        rm -f /tmp/uzh_api_server.pid
    fi
    
    # Kill web frontend
    if [ -f /tmp/uzh_web_frontend.pid ]; then
        WEB_PID=$(cat /tmp/uzh_web_frontend.pid)
        if ps -p $WEB_PID > /dev/null; then
            kill $WEB_PID
            print_status "Stopped web frontend (PID: $WEB_PID)"
        fi
        rm -f /tmp/uzh_web_frontend.pid
    fi
    
    # Kill any remaining processes
    pkill -f "main.py" || true
    pkill -f "next dev" || true
    
    print_status "System shutdown complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

main() {
    print_header
    
    case "${1:-full}" in
        "check")
            check_prerequisites
            ;;
        "install")
            check_prerequisites
            install_dependencies
            ;;
        "api")
            check_prerequisites
            start_api_server
            print_status "API server running. Press Ctrl+C to stop."
            wait
            ;;
        "web")
            check_prerequisites
            start_web_frontend
            print_status "Web frontend running. Press Ctrl+C to stop."
            wait
            ;;
        "eval")
            check_prerequisites
            run_evaluation_demo
            ;;
        "full"|*)
            check_prerequisites
            install_dependencies
            start_api_server
            start_web_frontend
            show_startup_summary
            run_evaluation_demo
            
            # Keep the script running
            print_status "System is running. Press Ctrl+C to stop all services."
            while true; do
                sleep 1
            done
            ;;
    esac
}

# Show usage if help requested
if [[ "${1}" == "-h" || "${1}" == "--help" || "${1}" == "help" ]]; then
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  full     Start complete system (default)"
    echo "  check    Check prerequisites only"
    echo "  install  Install dependencies only"
    echo "  api      Start API server only"
    echo "  web      Start web frontend only"
    echo "  eval     Run evaluation demo only"
    echo "  help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0           # Start complete system"
    echo "  $0 api       # Start only the API server"
    echo "  $0 eval      # Run only the evaluation demo"
    exit 0
fi

main "$@"
